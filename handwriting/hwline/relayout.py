"""Training-free fix for missing within-line dynamics: re-lay out a generated line's words.

A generated line is cut into its written words (the same column-gap segmentation the metric
uses), each word keeps its glyphs, and only its vertical offset and the gap before it change.
Targets are expressed in core heights, relative to the line's own mean baseline and mean gap,
so the generator's size, spacing level and style are preserved; only the variation changes.

Target trajectories:
  ident   no change (re-composition only; isolates cut/paste artefacts)
  jitter  iid baseline offsets and iid log-normal gap ratios with the real marginal spread,
          no trend (the naive augmentation control)
  donor   baseline residuals, drift slope and relative gaps copied from a random real train
          line with the same number of words (population model, writer-agnostic)
  style   slope, residual SD and gap CV taken from the style reference line (a real line of
          the target writer, which the generator was given as its input)
"""
from __future__ import annotations

import numpy as np

from hwline.descriptors import word_descriptors
from hwline.segment import segment_words, written_tokens

OUTLIER = 0.45  # |baseline - line trend| in core heights treated as measurement error


def cut_words(gray: np.ndarray, transcript: str):
    """-> list of word dicts (ink column span, baseline, core height) or None."""
    toks = written_tokens(transcript)
    boxes, thr, _ = segment_words(gray, len(toks))
    if boxes is None:
        return None
    words = []
    for t, b in zip(toks, boxes):
        d = word_descriptors(gray, b, thr, t)
        if not d["n_ink"]:
            return None
        words.append({"text": t, "x0": int(d["ink_x0"]), "x1": int(d["ink_x1"]),
                      "base": d["baseline_y"] if d["v_baseline"] else np.nan,
                      "core": d["core_h"] if d["v_core"] else np.nan})
    return words


def line_geometry(words):
    """Core height, word centres in core units (centred), measured baseline deviations from
    the line's own linear fit (0 where invalid or outlying), and the gaps in pixels."""
    core = np.nanmedian([w["core"] for w in words])
    if not np.isfinite(core):
        return None
    xc = np.array([(w["x0"] + w["x1"]) / 2 for w in words]) / core
    xc = xc - xc.mean()
    b = np.array([w["base"] for w in words]) / core
    ok = np.isfinite(b)
    if ok.sum() >= 2:
        k, c = np.polyfit(xc[ok], b[ok], 1)
    else:
        k, c = 0.0, np.nanmedian(b) if ok.any() else 0.0
    dev = np.where(ok, b - (k * xc + c), 0.0)
    dev = np.where(np.abs(dev) > OUTLIER, 0.0, dev)
    gaps = np.array([words[i + 1]["x0"] - words[i]["x1"] for i in range(len(words) - 1)], float)
    return {"core": core, "xc": xc, "slope": k, "dev": dev, "gaps": gaps}


def trajectory_stats(base, xc, gaps):
    """Summary of a real line's dynamics, in core units: slope, detrended residual SD, gap CV."""
    ok = np.isfinite(base) & np.isfinite(xc)
    if ok.sum() < 4:
        return None
    k, c = np.polyfit(xc[ok], base[ok], 1)
    r = base[ok] - (k * xc[ok] + c)
    r = r[np.abs(r) <= OUTLIER]
    g = gaps[np.isfinite(gaps)]
    if len(g) < 2 or g.mean() <= 0:
        return None
    return {"slope": k, "res_sd": float(r.std()) if len(r) >= 3 else 0.0,
            "gap_cv": float(g.std() / g.mean())}


def target(kind, geo, rng, donor=None, style=None, pop=None):
    """-> (baseline offsets in core units relative to the line fit, gap ratios to the mean gap)."""
    n = len(geo["xc"])
    xc = geo["xc"]
    if kind == "ident":
        g = geo["gaps"]
        return geo["dev"], g / g.mean() if g.mean() > 0 else np.ones(n - 1)
    if kind == "jitter":
        dev = rng.normal(0, pop["base_sd"], n)
        s = np.sqrt(np.log1p(pop["gap_cv"] ** 2))
        return dev, rng.lognormal(0, s, n - 1)
    if kind == "donor":
        base, dxc, dg = donor
        ok = np.isfinite(base)
        k, c = np.polyfit(dxc[ok], base[ok], 1)
        r = np.where(ok, base - (k * dxc + c), 0.0)
        r = np.where(np.abs(r) > OUTLIER, 0.0, r)
        g = np.where(np.isfinite(dg) & (dg > 0), dg, np.nan)
        g = np.where(np.isfinite(g), g, np.nanmean(g))
        return k * xc + r, g / g.mean()
    if kind == "style":
        dev = style["slope"] * xc + rng.normal(0, style["res_sd"], n)
        s = np.sqrt(np.log1p(style["gap_cv"] ** 2))
        return dev, rng.lognormal(0, s, n - 1)
    raise ValueError(kind)


def compose(gray, words, geo, dev, gap_ratio, min_gap=2):
    """Paste each word's ink columns at its new offset; the line's mean gap is kept."""
    h = gray.shape[0]
    bg = float(np.median(gray))
    ink_lvl = float(np.percentile(gray, 1))
    if bg - ink_lvl < 10:
        return None
    gaps = geo["gaps"]
    mean_gap = gaps.mean() if len(gaps) else 0.0
    ratio = gap_ratio * (len(gap_ratio) / gap_ratio.sum()) if len(gap_ratio) else gap_ratio
    new_gaps = np.maximum(mean_gap * ratio, min_gap)
    # Vertical shift: replace measured deviation by the target one (keeps the line's own fit).
    dy = (dev - geo["dev"]) * geo["core"]
    pad = int(np.ceil(np.abs(dy).max())) + 2
    widths = [w["x1"] - w["x0"] for w in words]
    x0 = words[0]["x0"]
    W = int(x0 + sum(widths) + new_gaps.sum() + (gray.shape[1] - words[-1]["x1"])) + 1
    canvas = np.full((h + 2 * pad, W), bg, np.float32)
    x = float(x0)
    for i, w in enumerate(words):
        slab = gray[:, w["x0"]:w["x1"]].astype(np.float32)
        a = np.clip((bg - slab) / (bg - ink_lvl), 0, 1)
        y = pad + int(round(dy[i]))
        xi = int(round(x))
        reg = canvas[y:y + h, xi:xi + slab.shape[1]]
        reg[:] = np.minimum(reg, bg * (1 - a[:, :reg.shape[1]]) + slab[:, :reg.shape[1]] * a[:, :reg.shape[1]])
        x += widths[i] + (new_gaps[i] if i < len(new_gaps) else 0)
    # Crop rows back to the ink extent plus the original top/bottom margins.
    rows = np.nonzero((canvas < bg - 0.5 * (bg - ink_lvl)).any(1))[0]
    orow = np.nonzero((gray < bg - 0.5 * (bg - ink_lvl)).any(1))[0]
    top_m, bot_m = orow[0], h - orow[-1] - 1
    r0, r1 = max(rows[0] - top_m, 0), min(rows[-1] + 1 + bot_m, canvas.shape[0])
    return np.clip(canvas[r0:r1], 0, 255).astype(np.uint8)


def relayout(gray, transcript, kind, rng, **kw):
    words = cut_words(gray, transcript)
    if words is None or len(words) < 2:
        return None
    geo = line_geometry(words)
    if geo is None:
        return None
    if kind == "donor":
        kw["donor"] = kw["pick_donor"](len(words))
        if kw["donor"] is None:
            return None
    dev, gr = target(kind, geo, rng, donor=kw.get("donor"), style=kw.get("style"), pop=kw.get("pop"))
    return compose(gray, words, geo, dev, gr)
