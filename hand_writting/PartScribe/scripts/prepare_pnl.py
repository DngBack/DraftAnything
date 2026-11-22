"""Pack the PNL uppercase handwriting release (handwriting_word_v1.0) at 32 px for PartScribe.

The release has 7,600 word crops from 19 real writers, 96% uppercase, covering all 89
Vietnamese capitals. The VN training data has 0 images of uppercase Ễ/Ê/Ộ.

Preprocessing matches the VATr-pp merge of the same release and the VN word images:
RGB -> L, then resize the (already tight) crop to height 32 keeping the aspect (BOX when
shrinking, else BICUBIC). Measured at 32 px over 1,500 crops:
  top / bottom margin 0.87 / 0.92 px, stroke ink p90 0.962, median aspect 1.5,
versus 0.87 / 0.90 px, 0.986 and 1.5 for VNOnDB train.

Filtering:
  * holdout protocol v1, stricter: drop any transcript with a held-out character in
    EITHER case (the NFD letter tokens are shared across case), as for HAT2023;
  * drop flagged crops ('overwritten', 'text_uncertain');
  * writer split: TEST_WRITERS (5 of 19, drawn once with seed 742) never enter training.
    They are the real-writer uppercase test set (C4).

    python scripts/prepare_pnl.py --release /home/giaphk/hand_writing_data/releases/handwriting_word_v1.0 \
        --out data/pnl32.npz
"""
import argparse
import hashlib
import json
import random
import sys
import unicodedata as ud
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.holdout import held_chars  # noqa: E402

HEIGHT = 32
SPLIT_SEED = 742
N_TEST_WRITERS = 5
PNL_FIRST_WRITER = 20000   # PartScribe writer ids for PNL: 20000 + index of NVxx in sorted order


def resize32(path):
    with Image.open(path) as im:
        im = im.convert('L')
        w, h = im.size
        nw = max(1, round(w * HEIGHT / h))
        return np.asarray(im.resize((nw, HEIGHT), Image.BOX if h > HEIGHT else Image.BICUBIC), np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('data/pnl32.npz'))
    a = ap.parse_args()

    rel = a.release
    sums = (rel / 'SHA256SUMS').read_bytes()
    samples = [json.loads(line) for line in (rel / 'samples.jsonl').read_text().splitlines() if line.strip()]
    writers = sorted({s['writer_id'] for s in samples})
    wid = {w: PNL_FIRST_WRITER + k for k, w in enumerate(writers)}
    test_writers = sorted(random.Random(SPLIT_SEED).sample(writers, N_TEST_WRITERS))

    held = set(held_chars('vn'))
    held |= {c.upper() for c in held}
    kept, dropped = [], {'heldout_any_case': 0, 'flagged': 0}
    for s in samples:
        t = ud.normalize('NFC', s['transcription'])
        if held & set(t):
            dropped['heldout_any_case'] += 1
            continue
        if set(s.get('flags') or []) & {'overwritten', 'text_uncertain'}:
            dropped['flagged'] += 1
            continue
        kept.append((s, t))

    cols, starts, widths, texts, wr, ids, split = [], [], [], [], [], [], []
    x = 0
    for s, t in kept:
        arr = resize32(rel / s['image'])
        cols.append(arr)
        starts.append(x); widths.append(arr.shape[1]); x += arr.shape[1]
        texts.append(t); wr.append(wid[s['writer_id']]); ids.append(s['id'])
        split.append('test' if s['writer_id'] in test_writers else 'train')
    pixels = np.concatenate(cols, axis=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, pixels=pixels, starts=np.array(starts), widths=np.array(widths))
    meta = {'release': str(rel), 'release_sha256sums_sha256': hashlib.sha256(sums).hexdigest(),
            'preprocess': 'RGB->L, resize to height 32 keeping aspect (BOX if shrinking else BICUBIC)',
            'filter': 'holdout v1 in either case; flags overwritten/text_uncertain', 'dropped': dropped,
            'writer_map': wid, 'test_writers': test_writers, 'split_seed': SPLIT_SEED,
            'text': texts, 'writer': wr, 'id': ids, 'split': split}
    a.out.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False))
    n_tr = split.count('train')
    caps = {c for t, sp in zip(texts, split) if sp == 'train' for c in t if c.isupper()}
    print(json.dumps({'kept': len(kept), 'train': n_tr, 'test': len(kept) - n_tr, 'dropped': dropped,
                      'test_writers': test_writers, 'train_capitals': len(caps),
                      'sha256_npz': hashlib.sha256(a.out.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
