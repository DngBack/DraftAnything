"""Pack IAM word images at a fixed height (default 64 px) from the original scans.

Word membership and split follow the VATr Groundtruth lists used by every HTG
paper (gan.iam.tr_va.gt.filter27 / gan.iam.test.gt.filter27). Boxes come from
ascii/words.txt (form coordinates) and are cut from the line images, offset by
the line box in ascii/lines.txt (the local forms/ folder is incomplete, lines/
is complete). The background is whitened with IAM's per-word grey threshold.

    python scripts/prepare_iam.py --iam /home/bachdx2/DraftAnything/handwriting/data/iam \
        --groundtruth ../VATr-pp/Groundtruth --out data/iam64.npz --height 64
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iam', type=Path, required=True)
    ap.add_argument('--groundtruth', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--height', type=int, default=64)
    a = ap.parse_args()
    wanted = {}
    for split, name in [('train_val', 'gan.iam.tr_va.gt.filter27'), ('test', 'gan.iam.test.gt.filter27')]:
        for line in (a.groundtruth / name).read_text(encoding='utf-8').splitlines():
            head, _, text = line.partition(' ')
            writer, wid = head.split(',')
            wanted[wid] = (split, head, writer, text)
    boxes = defaultdict(list)
    for row in open(a.iam / 'ascii' / 'words.txt', encoding='utf-8'):
        if row.startswith('#'):
            continue
        f = row.rstrip('\n').split(' ')
        if f[0] in wanted:
            boxes['-'.join(f[0].split('-')[:2])].append((f[0], int(f[2]), *map(int, f[3:7])))
    line_box = {}
    for row in open(a.iam / 'ascii' / 'lines.txt', encoding='utf-8'):
        if not row.startswith('#'):
            f = row.split(' ')
            line_box[f[0]] = tuple(map(int, f[4:8]))
    cols, meta, starts, widths, pos = [], defaultdict(list), [], [], 0
    for form in sorted(boxes):
        cache = {}
        for wid, gray, x, y, w, h in sorted(boxes[form]):
            line = '-'.join(wid.split('-')[:3])
            if line not in cache:
                cache[line] = np.asarray(Image.open(a.iam / 'lines' / form.split('-')[0] / form / f'{line}.png').convert('L'))
            page = cache[line]
            lx, ly = line_box[line][:2]
            x0, y0 = max(0, x - lx), max(0, y - ly)
            crop = page[y0:y0 + h, x0:x0 + w].astype(np.float32)
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            crop[crop > gray] = 255
            nw = max(1, round(crop.shape[1] * a.height / crop.shape[0]))
            img = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((nw, a.height), Image.BILINEAR))
            split, head, writer, text = wanted[wid]
            cols.append(img)
            starts.append(pos)
            widths.append(nw)
            pos += nw
            for k, v in (('split', split), ('head', head), ('writer', writer), ('text', text)):
                meta[k].append(v)
    np.savez(a.out, pixels=np.concatenate(cols, 1), starts=np.array(starts), widths=np.array(widths))
    meta.update(height=a.height, missing=len(wanted) - len(starts))
    Path(str(a.out).replace('.npz', '.json')).write_text(json.dumps(meta, ensure_ascii=False))
    print(f'{len(starts)} words packed, {meta["missing"]} listed words without a usable box')


if __name__ == '__main__':
    main()
