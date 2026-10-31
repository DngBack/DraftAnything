"""Per-word geometric descriptors measured on original-resolution IAM line images.

All measurements are image-processing estimates with known failure modes; every
descriptor comes with a validity flag and the analysis only uses flagged-valid values.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

LOWER_CORE = set("acemnorsuvwxz")          # letters with no ascender/descender
ASC = set("bdfhklt") | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | set("0123456789")
DESC = set("gjpqy")
SLANT_ANGLES = np.arange(-45, 46, 1.0)


def ink_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    """IAM gives a per-line binarisation threshold; ink is darker than it."""
    return gray < threshold


def core_band(ink: np.ndarray, frac: float = 0.5) -> tuple[int, int] | None:
    """Longest run of rows whose ink count >= frac * max row count (x-height band)."""
    prof = ink.sum(1).astype(float)
    if prof.max() <= 0:
        return None
    on = prof >= frac * prof.max()
    best, cur, start = (0, -1, -1), 0, 0
    for i, v in enumerate(on):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best[0]:
                best = (cur, start, i)
        else:
            cur = 0
    return (best[1], best[2]) if best[0] > 0 else None


def _shear_columns(ys: np.ndarray, xs: np.ndarray, angle_deg: float, ref_y: float):
    """Shift each ink pixel horizontally so a stroke slanted by angle becomes vertical."""
    t = np.tan(np.deg2rad(angle_deg))
    return np.round(xs - (ref_y - ys) * t).astype(int)


def slant(ink: np.ndarray, min_cols: int = 4) -> float | None:
    """Vinciarelli-Luettin style: angle maximising sum of squared heights of
    columns containing one continuous stroke after deslanting. Positive = right lean."""
    ys, xs = np.nonzero(ink)
    if len(ys) < 30:
        return None
    ref = ys.mean()
    best_s, best_a, n_cols = -1.0, None, 0
    for a in SLANT_ANGLES:
        cx = _shear_columns(ys, xs, a, ref)
        cx -= cx.min()
        cnt = np.bincount(cx)
        ymin = np.full(cnt.size, 10**9); ymax = np.full(cnt.size, -1)
        np.minimum.at(ymin, cx, ys); np.maximum.at(ymax, cx, ys)
        cont = (cnt > 0) & (ymax - ymin + 1 == cnt)
        s = float((cnt[cont].astype(float) ** 2).sum())
        if s > best_s:
            best_s, best_a, n_cols = s, a, int(cont.sum())
    return float(best_a) if n_cols >= min_cols else None


def slant_gradient(gray: np.ndarray, ink: np.ndarray, max_abs: float = 50.0) -> float | None:
    """Independent slant estimate from edge orientation: on a stroke leaning by theta the
    intensity gradient is proportional to (1, tan theta), so theta = atan(gy/gx). Weighted
    (|g|^2) mean over edge pixels of near-vertical strokes. Positive = right lean."""
    if ink.sum() < 30:
        return None
    g = ndimage.gaussian_filter(255.0 - gray.astype(float), 1.0)
    gx, gy = ndimage.sobel(g, 1), ndimage.sobel(g, 0)
    mag2 = gx ** 2 + gy ** 2
    near = ndimage.binary_dilation(ink, iterations=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        th = np.degrees(np.arctan(gy / gx))
    sel = near & np.isfinite(th) & (np.abs(th) < max_abs) & (mag2 > 0.1 * mag2[near].max())
    if sel.sum() < 30:
        return None
    return float(np.average(th[sel], weights=mag2[sel]))


def stroke_width(ink: np.ndarray) -> float | None:
    if ink.sum() < 30:
        return None
    sk = skeletonize(ink)
    if sk.sum() < 10:
        return None
    dt = ndimage.distance_transform_edt(ink)
    return float(2.0 * np.mean(dt[sk]))


def word_descriptors(gray: np.ndarray, box: tuple[int, int, int, int], threshold: int,
                     text: str) -> dict:
    """box is (x0, y0, x1, y1) in line-image coordinates."""
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1]
    ink = ink_mask(crop, threshold)
    n_ink = int(ink.sum())
    n_core_letters = sum(ch in LOWER_CORE for ch in text)
    n_lower = sum(ch.islower() for ch in text)
    out = {"n_ink": n_ink, "ink_x0": np.nan, "ink_x1": np.nan, "bg": np.nan,
           "baseline_y": np.nan, "core_h": np.nan, "slant": np.nan, "stroke_w": np.nan,
           "slant_grad": np.nan,
           "v_baseline": False, "v_core": False, "v_slant": False, "v_slant_grad": False,
           "v_stroke": False}
    if n_ink == 0:
        return out
    cols = np.nonzero(ink.any(0))[0]
    out["ink_x0"], out["ink_x1"] = float(x0 + cols[0]), float(x0 + cols[-1] + 1)
    bgpx = crop[~ink]
    out["bg"] = float(bgpx.mean()) if bgpx.size else np.nan

    # Core band only trusted when the word has >= 2 lowercase letters and >= 1 x-height-only letter.
    band = core_band(ink)
    if band is not None and n_lower >= 2 and n_core_letters >= 1:
        top, bot = band
        h = bot - top + 1
        if h >= 4:
            out["baseline_y"] = float(y0 + bot + 1)
            out["core_h"] = float(h)
            out["v_baseline"] = out["v_core"] = True
    if sum(ch.isalnum() for ch in text) >= 2:
        s = slant(ink)
        if s is not None and abs(s) < 44:      # boundary of the search grid = failed fit
            out["slant"], out["v_slant"] = s, True
        sg = slant_gradient(crop, ink)
        if sg is not None:
            out["slant_grad"], out["v_slant_grad"] = sg, True
        sw = stroke_width(ink)
        if sw is not None:
            out["stroke_w"], out["v_stroke"] = sw, True
    return out


def content_features(text: str) -> dict:
    n = max(len(text), 1)
    f = {
        "n_chars": len(text),
        "n_lower": sum(c.islower() for c in text),
        "n_upper": sum(c.isupper() for c in text),
        "n_digit": sum(c.isdigit() for c in text),
        "n_punct": sum((not c.isalnum()) for c in text),
        "n_asc": sum(c in ASC for c in text),
        "n_desc": sum(c in DESC for c in text),
        "n_core": sum(c in LOWER_CORE for c in text),
        "frac_upper": sum(c.isupper() for c in text) / n,
        "has_desc": float(any(c in DESC for c in text)),
        "has_asc_lower": float(any(c in "bdfhklt" for c in text)),
        "first_upper": float(text[:1].isupper()),
        "has_i_j": float(any(c in "ij" for c in text)),
    }
    low = text.lower()
    for ch in "abcdefghijklmnopqrstuvwxyz":
        f[f"c_{ch}"] = low.count(ch)
    return f
