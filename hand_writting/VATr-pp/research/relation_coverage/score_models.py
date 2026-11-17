"""Score saved generations of several models with two independent metrics (HWD backbone, writer-ID net).

For each model and condition: paired distance to the real target, writer-retrieval top-1 among the
H1 writers, and paired contrasts (writer-cluster bootstrap). Real-data references give the scale:
same-writer/same-word vs same-writer/different-word vs other-writer, per metric.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words, HWDFeatures, ink_crop, to_h32  # noqa: E402
import writer_id  # noqa: E402

CONDS = ['SELF', 'COV', 'UNC', 'OTHER']
rng = random.Random(0)


def boot(byw, B=4000):
    ws = sorted(byw)
    est = float(np.mean([x for w in ws for x in byw[w]]))
    bs = [np.mean([x for w in (rng.choice(ws) for _ in ws) for x in byw[w]]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--wid', required=True)
    p.add_argument('--models', nargs='+', default=['vatr', 'emuru'])
    a = p.parse_args()
    dev = 'cuda'
    rows = load_words(a.data)
    byid = {r['word_id']: r for r in rows}
    sups = [json.loads(l) for l in open(a.run / 'supports.jsonl')]
    sups = [s for s in sups if s['draw'] == 0]
    H1W = sorted({s['writer'] for s in sups})
    hwd = HWDFeatures(dev)
    wid = writer_id.load(a.wid, dev)

    def feats(arrays):
        arrays = [to_h32(ink_crop(x)) for x in arrays]
        return hwd(arrays), writer_id.embed_arrays(wid, arrays, dev)

    # real words of H1 writers -> features, centroids (excluding the target instance at query time)
    real_rows = [r for r in rows if r['writer'] in H1W and r['text'].isalpha() and len(r['text']) >= 2]
    RH, RW = feats([np.array(Image.open(r['path']).convert('L')) for r in real_rows])
    ridx = {r['word_id']: i for i, r in enumerate(real_rows)}
    wr = np.array([r['writer'] for r in real_rows])
    cent = {w: (RH[wr == w].sum(0), RW[wr == w].sum(0), int((wr == w).sum())) for w in H1W}

    def retrieve(e, target_id, true_w):
        """top-1 writer retrieval with the writer-ID embedding (cosine), target instance left out."""
        i = ridx.get(target_id)
        best, bw = -9, None
        for w in H1W:
            s, n = cent[w][1], cent[w][2]
            if w == true_w and i is not None:
                s, n = s - RW[i], n - 1
            sim = float(torch.nn.functional.normalize(s / n, dim=0) @ e)
            if sim > best:
                best, bw = sim, w
        return bw == true_w

    # real-data reference scale (same writer same word / different word / other writer)
    ref = defaultdict(list)
    byw = defaultdict(lambda: defaultdict(list))
    for r in real_rows:
        byw[r['writer']][r['text']].append(r)
    for w in H1W:
        for t, inst in byw[w].items():
            if len(inst) < 2 or len(t) < 3:
                continue
            diff = [x for tt, xs in byw[w].items() if tt != t and len(tt) == len(t) for x in xs]
            ow = rng.choice([x for x in H1W if x != w])
            oth = byw[ow].get(t) or [x for tt, xs in byw[ow].items() if len(tt) == len(t) for x in xs]
            if not diff or not oth:
                continue
            i0, i1, i2, i3 = (ridx[x['word_id']] for x in (inst[0], inst[1], rng.choice(diff), rng.choice(oth)))
            for name, F, dist in (('hwd', RH, lambda u, v: float((u - v).norm())),
                                  ('wid', RW, lambda u, v: float(1 - u @ v))):
                ref[f'{name}_same_word'].append(dist(F[i0], F[i1]))
                ref[f'{name}_diff_word'].append(dist(F[i0], F[i2]))
                ref[f'{name}_other_writer'].append(dist(F[i0], F[i3]))
    ref = {k: float(np.mean(v)) for k, v in ref.items()}
    ref['n_pairs'] = len([1 for _ in range(0)]) or None
    real_top1 = np.mean([retrieve(RW[ridx[s['word_id']]], s['word_id'], s['writer']) for s in sups])
    print('REAL reference:', {k: round(v, 4) for k, v in ref.items() if v is not None},
          f'| real target words writer-top1 = {real_top1:.3f} among {len(H1W)} writers')

    summary = dict(reference=ref, real_target_top1=float(real_top1), writers=len(H1W), models={})
    for m in a.models:
        gdir = a.run / 'gen' / m
        if not gdir.exists():
            continue
        units = [s for s in sups if all((gdir / c / f"{s['word_id']}.png").exists() for c in CONDS)]
        res = {}
        for c in CONDS:
            GH, GW = feats([np.array(Image.open(gdir / c / f"{s['word_id']}.png").convert('L')) for s in units])
            for k, s in enumerate(units):
                i = ridx[s['word_id']]
                res[(s['word_id'], c)] = dict(
                    writer=s['writer'],
                    hwd=float((GH[k] - RH[i]).norm()),
                    wid=float(1 - GW[k] @ RW[i]),
                    top1=float(retrieve(GW[k], s['word_id'], s['writer'])))
        out = dict(units=len(units), means={}, contrasts={})
        for c in CONDS:
            out['means'][c] = {k: float(np.mean([res[(s['word_id'], c)][k] for s in units])) for k in ('hwd', 'wid', 'top1')}
        for name, (x, y) in {'join UNC-COV': ('UNC', 'COV'), 'writer OTHER-COV': ('OTHER', 'COV'),
                             'copy COV-SELF': ('COV', 'SELF')}.items():
            for k in ('hwd', 'wid', 'top1'):
                bw = defaultdict(list)
                for s in units:
                    bw[s['writer']].append(res[(s['word_id'], x)][k] - res[(s['word_id'], y)][k])
                out['contrasts'][f'{name} [{k}]'] = boot(bw)
        # Emuru line supports cannot always match letter coverage: re-test on units where the UNC line
        # covers every letter of the target (then UNC differs from COV only in join coverage)
        cfile = a.run / f'{m}_conditions.jsonl'
        if cfile.exists():
            st = {c['word_id']: c['stats'] for c in map(json.loads, open(cfile))}
            full = [s for s in units if st[s['word_id']]['UNC']['letter_cov'] == 1]
            out['n_unc_full_letters'] = len(full)
            for k in ('hwd', 'wid', 'top1'):
                bw = defaultdict(list)
                for s in full:
                    bw[s['writer']].append(res[(s['word_id'], 'UNC')][k] - res[(s['word_id'], 'COV')][k])
                out['contrasts'][f'join UNC-COV [{k}] UNC-full-letters'] = boot(bw)
        # utilisation of word-specific evidence: copy effect / real same-vs-different-word gap
        for k in ('hwd', 'wid'):
            gap = ref[f'{k}_diff_word'] - ref[f'{k}_same_word']
            out[f'copy_utilisation_{k}'] = out['contrasts'][f'copy COV-SELF [{k}]'][0] / gap
        summary['models'][m] = out
        print(f'\n=== {m}: {len(units)} (writer, word) units, {len({s["writer"] for s in units})} writers ===')
        for c in CONDS:
            mm = out['means'][c]
            print(f"  {c:5s} hwd {mm['hwd']:.3f}  wid(1-cos) {mm['wid']:.4f}  writer-top1 {mm['top1']:.3f}")
        for k, (e, lo, hi) in out['contrasts'].items():
            print(f'  {k:28s} {e:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        print(f"  copy utilisation: hwd {100 * out['copy_utilisation_hwd']:.1f}%  wid {100 * out['copy_utilisation_wid']:.1f}%")
    json.dump(summary, open(a.run / 'two_model_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
