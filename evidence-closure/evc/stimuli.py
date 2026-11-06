"""Counterfactual digit edits on page images (docs/protocol_percept_cf.md)."""
from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_LIB = "/usr/share/fonts/truetype/liberation/Liberation"
FONTS = tuple(f"{_LIB}{n}.ttf" for n in ("Serif-Regular", "Serif-Bold", "Sans-Regular", "Sans-Bold"))


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def swap_digit(text: str, pos: int, new: str) -> str:
    """Replace the pos-th digit of text, keeping separators and symbols."""
    k = -1
    out = []
    for c in text:
        if c.isdigit():
            k += 1
            if k == pos:
                c = new
        out.append(c)
    return "".join(out)


def _fit(text, font_path, w, h):
    """Largest font size whose ink fits a w x h box."""
    size = max(6, int(h * 1.6))
    while size > 5:
        f = ImageFont.truetype(font_path, size)
        l, t, r, b = f.getbbox(text)
        if r - l <= w * 1.02 and b - t <= h * 1.05:
            return f, (l, t, r, b)
        size -= 1
    f = ImageFont.truetype(font_path, 5)
    return f, f.getbbox(text)


def _draw(img, text, bbox, font_path):
    """Erase the word with the box's own background colour; draw text with the original ink colour, right and bottom aligned to the original ink."""
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    a = np.asarray(img).astype(int)
    crop = a[y0:y1, x0:x1].reshape(-1, 3)
    cols, counts = np.unique(crop, axis=0, return_counts=True)
    bg = cols[np.argmax(counts)]  # modal colour inside the box: the cell background
    # ink extent, searched slightly beyond the OCR box so no stroke survives the erase
    X0, Y0, X1, Y1 = max(0, x0 - 4), max(0, y0 - 2), min(a.shape[1], x1 + 4), min(a.shape[0], y1 + 2)
    diff = np.abs(a[Y0:Y1, X0:X1] - bg).sum(-1) > 60
    ys, xs = np.where(diff[max(0, y0 - Y0):y1 - Y0, :])
    if len(xs) == 0:
        ix0, ix1, iy0, iy1 = x0, x1, y0, y1
    else:
        ix0, ix1 = X0 + xs.min(), X0 + xs.max() + 1
        iy0, iy1 = y0 + ys.min(), y0 + ys.max() + 1
    region = np.abs(a[iy0:iy1, ix0:ix1] - bg).sum(-1)
    ink = tuple(int(v) for v in a[iy0:iy1, ix0:ix1][np.unravel_index(np.argmax(region), region.shape)]) if region.size else (0, 0, 0)
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([ix0 - 1, iy0 - 1, ix1, iy1], fill=tuple(int(v) for v in bg))
    f, (l, t, r, b) = _fit(text, font_path, ix1 - ix0, iy1 - iy0)
    d.text((ix1 - r, iy1 - b), text, font=f, fill=ink)
    return out


def blur_canvas(img, scale: float):
    """Downsample to scale, then back to the original size: detail is lost, token count is not."""
    if scale == 1.0:
        return img.copy()
    small = img.resize((max(28, round(img.width * scale)), max(28, round(img.height * scale))), Image.BICUBIC)
    return small.resize(img.size, Image.BICUBIC)


def restore(canvas, sharp, bboxes, pad: int = 2):
    """Paste the original pixels back inside each box."""
    out = canvas.copy()
    for b in bboxes:
        box = (max(0, int(b[0]) - pad), max(0, int(b[1]) - pad), min(sharp.width, int(b[2]) + pad + 1), min(sharp.height, int(b[3]) + pad + 1))
        out.paste(sharp.crop(box), box[:2])
    return out


def draw_box(img, bbox, pad: int = 4, width: int = 3):
    """Red outline around a cell, outside its ink."""
    out = img.copy()
    x0, y0, x1, y1 = bbox
    ImageDraw.Draw(out).rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(230, 0, 0), width=width)
    return out


def best_font(img, text, bbox):
    """Font whose re-rendering of the original word is closest to the original pixels."""
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    base = np.asarray(img.crop((x0, y0, x1, y1)).convert("L"), float)
    errs = [np.abs(np.asarray(_draw(img, text, bbox, p).crop((x0, y0, x1, y1)).convert("L"), float) - base).mean() for p in FONTS]
    return FONTS[int(np.argmin(errs))]


def edit(img, word_text, bbox, new_text, font_path=None) -> Image.Image:
    return _draw(img, new_text, bbox, font_path or best_font(img, word_text, bbox))
