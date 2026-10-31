"""Perception pilot (plan §4.3 as a human test): can readers tell real within-line dynamics
from manipulated ones, with glyphs held fixed?

Each real val1 line is cut into written words (XML boxes; punctuation attached to its word,
ink pixels in overlapping boxes go to the nearest box centre) and re-composed on a white
canvas. Every condition, including the reference, goes through the same re-composition, so
cut/paste artefacts cannot give the answer away.

  orig     words at their measured baseline and gaps (identity re-composition)
  shuffle  baseline residuals and inter-word gaps permuted within the line: same marginal
           variation, within-line dependence and drift destroyed
  detrend  per-line linear trend of the baseline removed, local residuals and gaps kept
  flat     every word on the line-mean baseline with the line-mean gap (word stitching)
  catch    iid vertical jitter of 0.6 core height (attention check, should be obvious)

Words without a valid baseline estimate, or more than OUTLIER core heights off the line's
fitted trend (mostly short-word measurement errors), ride on that trend.
Writes PNGs (height --out-h) and stimuli.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CONDS = ["orig", "shuffle", "detrend", "flat", "catch"]
OUTLIER = 0.45  # |baseline - line trend| in core heights


def group_words(texts, boxes):
    """XML words -> written-word groups (indices); punctuation joins its neighbour."""
    groups = []
    for i, t in enumerate(texts):
        alnum = any(c.isalnum() for c in t)
        if not alnum and groups:
            groups[-1].append(i)
        elif groups and not any(any(c.isalnum() for c in texts[j]) for j in groups[-1]):
            groups[-1].append(i)  # leading punctuation joins the first real word
        else:
            groups.append([i])
    return groups


def word_layers(gray, ink, boxes, groups):
    """Per group: (x0, y0, crop gray, crop mask) with each ink pixel owned by one box."""
    h, w = gray.shape
    cx = np.array([(b[0] + b[2]) / 2 for b in boxes])
    owner = np.full((h, w), -1, int)
    best = np.full((h, w), np.inf)
    xs = np.arange(w)[None, :].repeat(h, 0)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        sl = (slice(max(y0, 0), min(y1, h)), slice(max(x0, 0), min(x1, w)))
        d = np.abs(xs[sl] - cx[i])
        take = d < best[sl]
        best[sl][take] = d[take]
        owner[sl][take] = i
    layers = []
    for g in groups:
        m = np.isin(owner, g) & ink
        ys, xs_ = np.nonzero(m)
        if not len(ys):
            layers.append(None)
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs_.min(), xs_.max() + 1
        layers.append((x0, y0, gray[y0:y1, x0:x1], m[y0:y1, x0:x1]))
    return layers


def compose(layers, dy, x_new, h, pad):
    W = int(max(x + L[2].shape[1] for x, L in zip(x_new, layers))) + 2 * pad
    canvas = np.full((h + 2 * pad, W), 255, np.uint8)
    for L, d, x in zip(layers, dy, x_new):
        _, y0, g, m = L
        yy, xx = int(round(y0 + d)) + pad, int(round(x)) + pad
        yy = min(max(yy, 0), canvas.shape[0] - g.shape[0])
        sub = canvas[yy:yy + g.shape[0], xx:xx + g.shape[1]]
        sub[m] = np.minimum(sub[m], g[m])
    ys, xs = np.nonzero(canvas < 255)
    return canvas[max(ys.min() - pad, 0):ys.max() + pad + 1, :]


def build_line(rec, words, rng):
    texts, boxes = json.loads(rec.word_texts), json.loads(rec.word_boxes)
    gray = np.asarray(Image.open(rec.image_path).convert("L")).copy()
    ink = gray < rec.threshold
    gray[~ink] = 255
    groups = group_words(texts, boxes)
    layers = word_layers(gray, ink, boxes, groups)
    keep = [i for i, L in enumerate(layers) if L is not None]
    groups, layers = [groups[i] for i in keep], [layers[i] for i in keep]
    n = len(layers)
    wd = words.set_index("pos")
    base = np.full(n, np.nan)
    core = []
    for k, g in enumerate(groups):
        r = wd.loc[g[0]] if g[0] in wd.index else None
        if r is not None and bool(r.v_baseline):
            base[k] = r.baseline_y
            core.append(r.core_h)
    xc = np.array([L[0] + L[2].shape[1] / 2 for L in layers])
    ok = ~np.isnan(base)
    if n < 6 or ok.sum() < max(4, int(0.6 * n)) or len(core) < 3:
        return None
    med_core = float(np.median(core))
    slope, icpt = np.polyfit(xc[ok], base[ok], 1)
    # Baseline estimates on short words are sometimes far off; displaced in shuffle/flat
    # they would turn measurement error into real offsets. Outliers ride the trend instead.
    ok &= np.abs(base - (slope * xc + icpt)) <= OUTLIER * med_core
    if ok.sum() < max(4, int(0.6 * n)):
        return None
    slope, icpt = np.polyfit(xc[ok], base[ok], 1)
    trend = slope * xc + icpt
    base = np.where(ok, base, trend)
    left = np.array([L[0] for L in layers], float)
    width = np.array([L[2].shape[1] for L in layers], float)
    gaps = left[1:] - (left[:-1] + width[:-1])
    if (gaps < 0).mean() > 0.2:
        return None

    def place(g):
        x = [left[0]]
        for i in range(1, n):
            x.append(x[-1] + width[i - 1] + g[i - 1])
        return np.array(x)

    resid = base - base.mean()
    target = {
        "orig": (base, gaps),
        "shuffle": (base.mean() + rng.permutation(resid), rng.permutation(gaps)),
        "detrend": (base - (trend - trend.mean()), gaps),
        "flat": (np.full(n, base.mean()), np.full(n - 1, gaps.mean())),
        "catch": (base + rng.normal(0, 0.6 * med_core, n), gaps),
    }
    pad = int(1.0 * med_core)
    out = {}
    for c, (b, g) in target.items():
        out[c] = compose(layers, b - base, place(g), gray.shape[0], pad)
    info = {"n_words": n, "n_valid": int(ok.sum()), "med_core": med_core,
            "slope_pct": float(slope * 100),
            "base_sd_core": float(resid.std() / med_core),
            "gap_cv": float(gaps.std() / max(gaps.mean(), 1e-6))}
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val1")
    ap.add_argument("--n-lines", type=int, default=160)
    ap.add_argument("--out", type=Path, default=Path("data/perception"))
    ap.add_argument("--out-h", type=int, default=72, help="ink-box height of saved images")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    man = pd.read_csv("data/manifest.csv", dtype={"writer_id": str})
    man = man[(man.split == a.split) & (man.quality_flag == "ok")]
    words = pd.read_csv(f"data/words_{a.split}.csv.gz", dtype={"writer_id": str})
    wby = dict(tuple(words.groupby("line_id")))
    # At most 4 lines per writer so no writer dominates the pilot.
    man = man.sample(frac=1, random_state=a.seed).groupby("writer_id").head(4)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in man.itertuples():
        if len(rows) >= a.n_lines or rec.line_id not in wby:
            continue
        res = build_line(rec, wby[rec.line_id], rng)
        if res is None:
            continue
        imgs, info = res
        H = max(im.shape[0] for im in imgs.values())
        s = a.out_h * 2.2 / H  # same scale for every condition of a line
        for c, im in imgs.items():
            pil = Image.fromarray(im)
            pil = pil.resize((max(int(pil.width * s), 1), max(int(pil.height * s), 1)), Image.LANCZOS)
            pil.save(a.out / f"{rec.line_id}__{c}.png", optimize=True)
        rows.append({"line_id": rec.line_id, "writer_id": rec.writer_id,
                     "transcript": rec.transcript, **info})
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "stimuli.csv", index=False)
    print(json.dumps({"lines": len(df), "writers": int(df.writer_id.nunique()),
                      "median_words": float(df.n_words.median()),
                      "median_base_sd_core": round(float(df.base_sd_core.median()), 3),
                      "median_abs_slope": round(float(df.slope_pct.abs().median()), 3)}))


if __name__ == "__main__":
    main()
