"""Crop IAM word images (official writer-independent test forms) for the H1 relation-coverage study.

Output: <out>/words/<word_id>.png (height --height, default 32, aspect kept, background whitened by IAM gray level)
        <out>/words.csv  (word_id, writer, form, text, width32)
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iam', type=Path, default=Path('/home/bachdx2/DraftAnything/handwriting/data/iam'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', default='testset.txt')
    p.add_argument('--height', type=int, default=32)
    p.add_argument('--line-height', action='store_true',
                   help='keep the full line height (word x-range only): consistent scale across words, as in line images')
    a = p.parse_args()

    lines = {l.strip() for l in open(a.iam / 'task' / a.split) if l.strip()}
    forms_in_split = {'-'.join(l.split('-')[:2]) for l in lines}
    writer = {}
    for row in open(a.iam / 'ascii' / 'forms.txt'):
        if row.startswith('#'):
            continue
        f = row.split()
        writer[f[0]] = f[1]

    by_form = defaultdict(list)
    for row in open(a.iam / 'ascii' / 'words.txt'):
        if row.startswith('#'):
            continue
        f = row.rstrip('\n').split(' ')
        wid, status, gray, x, y, w, h = f[0], f[1], int(f[2]), *map(int, f[3:7])
        text = ' '.join(f[8:])
        line = '-'.join(wid.split('-')[:3])
        form = '-'.join(wid.split('-')[:2])
        if status != 'ok' or line not in lines or w <= 1 or h <= 1:
            continue
        by_form[form].append((wid, gray, x, y, w, h, text))

    # Only line images are fully extracted; word boxes are in form coordinates, so offset by the line box.
    line_box = {}
    for row in open(a.iam / 'ascii' / 'lines.txt'):
        if row.startswith('#'):
            continue
        f = row.split(' ')
        line_box[f[0]] = tuple(map(int, f[4:8]))

    (a.out / 'words').mkdir(parents=True, exist_ok=True)
    rows = []
    for form, items in sorted(by_form.items()):
        cache = {}
        for wid, gray, x, y, w, h, text in items:
            line = '-'.join(wid.split('-')[:3])
            if line not in cache:
                cache[line] = np.array(Image.open(
                    a.iam / 'lines' / form.split('-')[0] / form / f'{line}.png').convert('L'))
            page = cache[line]
            lx, ly, _, _ = line_box[line]
            x0, y0 = max(0, x - lx), max(0, y - ly)
            crop = (page[:, x0:x0 + w] if a.line_height else page[y0:y0 + h, x0:x0 + w]).astype(np.float32)
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            crop[crop > gray] = 255  # IAM-provided binarisation threshold: whiten background only
            img = Image.fromarray(crop.astype(np.uint8))
            nw = max(1, round(img.size[0] * a.height / img.size[1]))
            img = img.resize((nw, a.height), Image.BILINEAR)
            img.save(a.out / 'words' / f'{wid}.png')
            rows.append((wid, writer[form], form, text, nw))
    with open(a.out / 'words.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['word_id', 'writer', 'form', 'text', 'width32'])
        w.writerows(rows)
    print(f'forms {len(by_form)} (split forms {len(forms_in_split)}), words {len(rows)}, '
          f'writers {len({r[1] for r in rows})}')


if __name__ == '__main__':
    main()
