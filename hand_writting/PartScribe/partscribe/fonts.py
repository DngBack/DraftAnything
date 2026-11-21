"""Multi-font glyph rendering with quality control and a leakage-safe font split.

Normalisation is per font, never per glyph: every character of one font goes
through the same scale and offset, so relative size (case) and mark positions
are preserved. The scale fits the union of all letter bounding boxes into the
canvas.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .charset import composition_chain, glyph_chars, parts

REF_SIZE = 160          # point size used before rescaling
CANVAS = 96
PAD = 2
MIN_MARK_INK = 4.0   # a mark must change >= 4 full pixels of a 96x96 glyph
# Families rejected after looking at the QC contact sheet (glyph defects the
# automatic checks miss). Reason is recorded in the bank metadata.
VISUAL_DENYLIST = {'intelonemono': 'visual_qc:garbled stacked marks (ễ) and ỷ'}
SPLITS = (('train', 0.75), ('val', 0.10), ('test', 0.15))


def group_key(record):
    """Families sharing a first name word (Noto*, Roboto*, Playwrite*) stay in one split."""
    return (record['name'] or record['family']).split()[0].lower()


def split_of(record, seed='partscribe-v1'):
    h = int(hashlib.sha256(f'{seed}:{group_key(record)}'.encode()).hexdigest()[:8], 16) / 2 ** 32
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if h < acc:
            return name
    return SPLITS[-1][0]


def _raw(font, ch):
    """Render one char at REF_SIZE on a big canvas, anchored at a fixed baseline origin."""
    W = REF_SIZE * 4
    im = Image.new('L', (W, W), 0)
    ImageDraw.Draw(im).text((W // 4, W // 2), ch, font=font, fill=255, anchor='ls')
    return np.asarray(im)


def render_font(path, chars, canvas=CANVAS):
    """Return (uint8 [len(chars), canvas, canvas] with ink=255, info dict)."""
    font = ImageFont.truetype(str(path), REF_SIZE)
    raws = [_raw(font, c) for c in chars]
    letters = [i for i, c in enumerate(chars) if c.isalpha()]
    boxes = []
    for i in letters:
        ys, xs = np.nonzero(raws[i] > 32)
        if len(xs):
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not boxes:
        raise ValueError('no ink')
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    scale = (canvas - 2 * PAD) / max(x1 - x0, y1 - y0)
    out = np.zeros((len(chars), canvas, canvas), np.uint8)
    for i, r in enumerate(raws):
        crop = Image.fromarray(r[y0:y1, x0:x1])
        w, h = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
        small = np.asarray(crop.resize((w, h), Image.LANCZOS))
        ox, oy = (canvas - w) // 2, (canvas - h) // 2
        out[i, oy:oy + h, ox:ox + w] = small
    return out, {'scale': scale, 'union_box': [int(x0), int(y0), int(x1), int(y1)]}


def _mask_iou(a, b):
    def norm(x):
        ys, xs = np.nonzero(x > 96)
        if not len(xs):
            return np.zeros((24, 24), bool)
        c = x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return np.asarray(Image.fromarray(c).resize((24, 24))) > 96
    a, b = norm(a), norm(b)
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def quality(glyphs, chars):
    """Reasons to reject a font, or [] when it is usable."""
    idx = {c: i for i, c in enumerate(chars)}
    ink = (glyphs > 96).reshape(len(chars), -1).sum(1)
    bad = []
    empty = [c for c in chars if ink[idx[c]] == 0]
    if empty:
        bad.append('empty:' + ''.join(empty[:10]))
    # Every composition step must change the glyph (fonts that drop accents fail).
    nomark, min_diff = [], float('inf')
    for c in chars:
        if not parts(c)[1]:
            continue
        chain, _ = composition_chain(c)
        for prev, cur in zip(chain, chain[1:]):
            # Absolute grey-level change, not added ink: in í/ỉ the mark replaces
            # the dot of i. Unit: fully-inked pixel equivalents.
            diff = np.abs(glyphs[idx[cur]].astype(int) - glyphs[idx[prev]].astype(int)).sum() / 255
            min_diff = min(min_diff, diff)
            if diff < MIN_MARK_INK:
                nomark.append(c)
                break
    if nomark:
        bad.append('mark_missing:' + ''.join(nomark[:10]))
    # Caps-only / small-caps fonts: lowercase shaped like uppercase.
    ious = [_mask_iou(glyphs[idx[l]], glyphs[idx[l.upper()]]) for l in 'abdefghijklmnqrty']
    if np.median(ious) > 0.75:
        bad.append(f'caps_like:{np.median(ious):.2f}')
    return bad, min_diff


def _job(args):
    path, chars = args
    try:
        g, info = render_font(path, chars)
        bad, min_diff = quality(g, chars)
        return g, {**info, 'min_mark_change': round(float(min_diff), 2)}, bad
    except Exception as e:  # unreadable or unrenderable font
        return None, {}, [f'error:{type(e).__name__}:{e}'[:120]]


def build_glyph_bank(manifest_path, font_dir, out_path, workers=16, limit=None):
    """Render all fully-covering fonts; save glyphs + per-font split/QC to one .npz + .json."""
    man = json.loads(Path(manifest_path).read_text())
    recs = [r for r in man['fonts'] if r['full_coverage']]
    recs.sort(key=lambda r: r['file'])
    if limit:
        recs = recs[:limit]
    chars = glyph_chars()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, [(Path(font_dir) / r['file'], chars) for r in recs], chunksize=4))
    keep, meta = [], []
    for r, (g, info, bad) in zip(recs, res):
        if r['family'] in VISUAL_DENYLIST:
            bad = bad + [VISUAL_DENYLIST[r['family']]]
        if g is not None and Path(font_dir, r['file']).exists():
            assert hashlib.sha256(Path(font_dir, r['file']).read_bytes()).hexdigest() == r['sha256']
        m = {'file': r['file'], 'family': r['family'], 'name': r['name'], 'category': r['category'],
             'group': group_key(r), 'split': split_of(r), 'rejected': bad, **info}
        meta.append(m)
        if not bad:
            m['index'] = len(keep)
            keep.append(g)
    glyphs = np.stack(keep)
    np.savez_compressed(out_path, glyphs=glyphs, chars=np.array(chars))
    meta_path = Path(str(out_path).replace('.npz', '.json'))
    meta_path.write_text(json.dumps({'chars': ''.join(chars), 'canvas': CANVAS, 'fonts': meta},
                                    ensure_ascii=False, indent=1))
    return glyphs, meta


def load_glyph_bank(npz_path):
    z = np.load(npz_path)
    meta = json.loads(Path(str(npz_path).replace('.npz', '.json')).read_text())
    fonts = [m for m in meta['fonts'] if not m['rejected']]
    fonts.sort(key=lambda m: m['index'])
    return z['glyphs'], list(meta['chars']), fonts
