"""Audit packed SpiS HDF5 and create leakage-filtered VATr manifests (no image copy)."""
import argparse
import csv
import hashlib
import json
import pickle
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def source(wid, test=False):
    # Verified against SpiS-GAN's merge documentation; test IDs are independent.
    if test:
        return 'test_unknown_source'
    return 'VNOnDB' if wid < 106 else ('ViHTGen' if wid < 295 else 'HAT2023')


def ctc_steps(width):
    return width // 4 + 1  # exact geometry of this repository's 32px OCR


def describe(values):
    return {k: float(v) for k, v in zip(('min', 'p50', 'p90', 'p99', 'max'), np.percentile(values, [0, 50, 90, 99, 100]))}


def scan(path, is_test=False):
    with h5py.File(path, 'r') as f:
        meta = {k: f[k][:] for k in ('img_seek_idxs', 'img_lens', 'lbs', 'lb_seek_idxs', 'lb_lens', 'wids')}
        n = len(meta['wids'])
        assert f['imgs'].shape[0] == 32, 'Only the original 32px data is supported'
        for prefix, end in [('img', f['imgs'].shape[1]), ('lb', len(meta['lbs']))]:
            starts, lengths = meta[prefix + '_seek_idxs'], meta[prefix + '_lens']
            assert len(starts) == len(lengths) == n
            assert np.all(lengths > 0) and starts[0] == 0
            assert np.all(starts[1:] == starts[:-1] + lengths[:-1])
            assert starts[-1] + lengths[-1] == end
        pixels = f['imgs'][:]
    records, summaries, counts = [], {}, {}
    for i, wid in enumerate(meta['wids']):
        a, l = int(meta['lb_seek_idxs'][i]), int(meta['lb_lens'][i])
        raw = ''.join(chr(int(c)) for c in meta['lbs'][a:a+l])
        text = unicodedata.normalize('NFC', raw)
        a, w = int(meta['img_seek_idxs'][i]), int(meta['img_lens'][i])
        im = np.ascontiguousarray(pixels[:, a:a+w])
        digest = hashlib.sha256(w.to_bytes(4, 'little') + im.tobytes()).hexdigest()
        src = source(int(wid), is_test)
        required = len(text) + sum(x == y for x, y in zip(text, text[1:]))
        invalid = ('empty' if not text.strip() else
                   'control' if any(unicodedata.category(c).startswith('C') for c in text) else
                   'flat' if im.max() == im.min() else
                   'ctc' if ctc_steps(w) < required else '')
        records.append(dict(index=i, writer=int(wid), source=src, text=text, width=w,
                            hash=digest, invalid=invalid))
        stats = summaries.setdefault(src, Counter())
        stats.update(samples=1, nfc_changed=int(raw != text), all_upper=int(text.isupper()),
                     with_space=int(any(c.isspace() for c in text)), invalid=int(bool(invalid)),
                     top_edge_ink=int((im[0] < 128).mean() > .02),
                     bottom_edge_ink=int((im[-1] < 128).mean() > .02))
        if invalid:
            stats['invalid_' + invalid] += 1
        counts.setdefault(src, Counter()).update(text)
    for src, stats in summaries.items():
        selected = [r for r in records if r['source'] == src]
        stats = summaries[src] = dict(stats)
        stats['writers'] = len(set(r['writer'] for r in selected))
        stats['samples_per_writer'] = describe(list(Counter(r['writer'] for r in selected).values()))
        stats['width'] = describe([r['width'] for r in selected])
        stats['label_length'] = describe([len(r['text']) for r in selected])
        stats['chars'] = len(counts[src])
        stats['over_192px'] = sum(r['width'] > 192 for r in selected)
    return records, summaries, counts


def prepare(data, output, glyph_path, seed=742, val_fraction=.1):
    output.mkdir(parents=True, exist_ok=True)
    train, train_stats, train_counts = scan(data / 'train.h5')
    test, test_stats, test_counts = scan(data / 'test.h5', True)
    rng = np.random.default_rng(seed)
    val_writers = []
    for src in ('VNOnDB', 'ViHTGen'):
        writers = sorted({r['writer'] for r in train if r['source'] == src})
        val_writers.extend(int(x) for x in rng.choice(writers, max(1, round(len(writers)*val_fraction)), replace=False))
    val_writers = set(val_writers)
    # Evaluation has precedence. Deduplicate on pixels, even if labels disagree.
    labels_by_hash = defaultdict(set)
    for r in train + test:
        labels_by_hash[r['hash']].add(r['text'])
    conflict_hashes = {h for h, labels in labels_by_hash.items() if len(labels) > 1}
    conflicts = [r for r in train + test if r['hash'] in conflict_hashes]
    (output/'ambiguous_labels.json').write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding='utf-8')
    quarantined = Counter()
    occupied = {}
    duplicate_stats = Counter()
    duplicate_pairs = Counter()
    kept = {}
    groups = [('test', test), ('val', [r for r in train if r['writer'] in val_writers]),
              ('train', [r for r in train if r['writer'] not in val_writers and r['writer'] < 295]),
              ('hat_ocr', [r for r in train if r['writer'] >= 295])]
    for name, records in groups:
        kept[name] = []
        for r in records:
            if r['invalid']:
                continue
            if r['hash'] in conflict_hashes:
                quarantined[name] += 1
                continue
            if r['hash'] in occupied:
                duplicate_stats[name] += 1
                owner, label = occupied[r['hash']]
                duplicate_pairs[f'{owner}->{name}'] += 1
                continue
            occupied[r['hash']] = (name, r['text'])
            kept[name].append(r)
    # Train-only alphabet and corpus: never select model support from test labels.
    charset = ''.join(sorted(set(''.join(r['text'] for name in ('train', 'hat_ocr') for r in kept[name])) | {' '}))
    with open(glyph_path, 'rb') as f:
        symbols = pickle.load(f)
    glyphs = {int(s['idx'][0]): s['mat'] for s in symbols}
    missing = [c for c in charset if ord(c) not in glyphs]
    if missing:
        raise ValueError(f'Missing Unifont glyphs: {missing!r}')
    selected = [dict(idx=[ord(c)], mat=glyphs[ord(c)]) for c in charset]
    with open(output / 'unifont.pickle', 'wb') as f:
        pickle.dump(selected, f)
    (output / 'charset.json').write_text(json.dumps({'alphabet': charset, 'normalization': 'NFC'}, ensure_ascii=False, indent=2), encoding='utf-8')
    corpus = [r['text'] for name in ('train', 'hat_ocr') for r in kept[name] if len(r['text']) <= 20]
    (output / 'corpus.txt').write_text('\n'.join(corpus) + '\n', encoding='utf-8')
    manifest = dict(version=1, seed=seed, source_train=str((data/'train.h5').resolve()),
                    source_test=str((data/'test.h5').resolve()), alphabet=charset,
                    glyph_path=str((output/'unifont.pickle').resolve()),
                    glyph_sha256=hashlib.sha256((output/'unifont.pickle').read_bytes()).hexdigest(),
                    val_writers=sorted(val_writers),
                    source_files={name: {'size': (data/name).stat().st_size, 'mtime_ns': (data/name).stat().st_mtime_ns} for name in ('train.h5', 'test.h5')},
                    indices={name: [r['index'] for r in records] for name, records in kept.items()})
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    report = dict(raw={'train': train_stats, 'test': test_stats},
                  retained={name: len(records) for name, records in kept.items()},
                  retained_writers={name: len(set(r['writer'] for r in records)) for name, records in kept.items()},
                  duplicate_images_removed=dict(duplicate_stats), duplicate_pairs=dict(duplicate_pairs),
                  ambiguous_image_groups=len(conflict_hashes), ambiguous_samples_quarantined=dict(quarantined), charset_size=len(charset),
                  hat_only_chars=sorted(set(''.join(r['text'] for r in kept['hat_ocr'])) - set(''.join(r['text'] for r in kept['train']))),
                  retained_char_counts={name: dict(Counter(''.join(r['text'] for r in records))) for name, records in kept.items()},
                  unseen_eval_chars={name: sorted(set(''.join(r['text'] for r in kept[name]))-set(charset)) for name in ('val', 'test')},
                  caveats=['HAT2023 IDs are per-image pseudo IDs; excluded from style/writer training.',
                           'Exact pixel deduplication does not detect resized/near duplicates or identify HAT writers.',
                           'Test IDs are local; numeric overlap with train does not imply shared writers.',
                           'Edge-ink also responds to dark backgrounds; it is not proof of lost diacritics.',
                           'Input images are already 32px; upsampling cannot restore lost diacritics.'])
    (output/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output/'characters.csv').open('w', encoding='utf-8', newline='') as f:
        names = list(train_counts) + list(test_counts)
        counters = {**train_counts, **test_counts}
        writer = csv.writer(f)
        writer.writerow(['character', 'codepoint'] + names)
        for c in sorted(set().union(*(set(v) for v in counters.values()))):
            writer.writerow([c, f'U+{ord(c):04X}'] + [counters[k][c] for k in names])
    # Deterministic examples per source, showing unmodified source pixels.
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new('RGB', (1000, 900), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, src in enumerate(('VNOnDB', 'ViHTGen', 'HAT2023')):
        draw.text((col*330+5, 5), src, fill='black', font=font)
        options = [r for r in train if r['source'] == src and not r['invalid']]
        with h5py.File(data/'train.h5', 'r') as f:
            for row, r in enumerate(rng.choice(options, 9, replace=False)):
                i = r['index']; a = int(f['img_seek_idxs'][i]); w = r['width']
                im = Image.fromarray(f['imgs'][:, a:a+w])
                im = im.resize((min(w*2, 315), 64))
                canvas.paste(im, (col*330+5, row*95+30))
                draw.text((col*330+5, row*95+95), f"{i}: {r['text']}", fill='black', font=font)
    canvas.save(output/'contact_sheet.png')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('files/vietnamese'))
    p.add_argument('--glyphs', type=Path, default=Path('../VATr/files/unifont.pickle'))
    p.add_argument('--seed', type=int, default=742)
    args = p.parse_args()
    prepare(args.data, args.output, args.glyphs, args.seed)
