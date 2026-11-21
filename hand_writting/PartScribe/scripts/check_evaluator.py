"""Gate for an evaluator OCR: it must read marks from the image, not from a word prior.

Erase test: real val images that contain a held-out character whose tone mark sits BELOW
the base (ẹ) or ABOVE it (all others); erase the mark region (bottom / top 6 px) and count
how often the OCR still outputs the character. A mark-faithful OCR should almost never do so.
Also reports its CER on the full val split (real images).

    python scripts/check_evaluator.py --ocr runs/eval_ocr_vn_local/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.datasets import load_vn, to_ink  # noqa: E402
from partscribe.evaluator import Codec, align, load_ocr  # noqa: E402
from partscribe.holdout import held_chars  # noqa: E402
from partscribe.runtime import VATR  # noqa: E402
from scripts.eval_holdout import ocr_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ocr', type=Path, required=True)
    ap.add_argument('--vn-data', default='data/vn')
    ap.add_argument('--band', type=int, default=6)
    a = ap.parse_args()
    dev = 'cuda'
    ocr, codec = load_ocr(a.ocr, dev), Codec()
    ds = load_vn(str(VATR / 'files/vietnamese/manifest.json'), a.vn_data, 'val', apply_holdout=False)
    held = held_chars('vn')
    out = {}
    for name, chars, rows in [('dot_below', 'ẹ', slice(-a.band, None)), ('above', ''.join(sorted(held - {'ẹ'})), slice(0, a.band))]:
        idx = [i for i, t in enumerate(ds.texts) if set(chars) & set(t)]
        imgs = [to_ink(ds.image(i)) for i in idx]
        cut = [im.clone() for im in imgs]
        for im in cut:
            im[rows] = 0
        h0, h1 = ocr_images(ocr, codec, imgs, dev), ocr_images(ocr, codec, cut, dev)
        hit = lambda hs: sum(any(c in h for c in set(chars) & set(ds.texts[i])) for i, h in zip(idx, hs))
        out[name] = {'n': len(idx), 'read_intact': hit(h0), 'read_after_erase': hit(h1)}
    ed = n = 0
    hyp = ocr_images(ocr, codec, [to_ink(ds.image(i)) for i in range(len(ds.texts))], dev)
    for t, h in zip(ds.texts, hyp):
        ed += align(t, h)[1]; n += len(t)
    out['val_cer'] = ed / n
    print(json.dumps({'ocr': str(a.ocr), **out}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
