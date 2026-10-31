"""Why do line-level generators lack within-line dynamics? Three data checks.

1. Is the dynamics a writer property at all? Variance of per-line stats explained by writer
   and by form (one-way random-effects share) on real train+val1 lines. Slope sharing a form
   but not a writer points at paper/scan skew rather than handwriting.
2. Does the style pathway carry it? Spearman correlation between the style reference line's
   stats and the output line's stats, for real (another real line of the same writer as the
   'output') and for each generator. CI by writer bootstrap.
3. Can TS-GAN's discriminator see it? Horizontal receptive field of its patch heads vs the
   width of the lines it is trained on (analytic, from model/discriminator_ap.py).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hwline.profile import per_line_stats

STATS = ["slope", "base_sd", "gap_cv"]


def group_share(df, col, group):
    """One-way random-effects share of variance between groups (ICC(1)), groups with >= 2 lines."""
    d = df[[col, group]].dropna()
    d = d[d.groupby(group)[col].transform("size") >= 2]
    g = d.groupby(group)[col]
    k = len(g)
    n0 = (len(d) - (g.size() ** 2).sum() / len(d)) / (k - 1)
    msb = (g.size() * (g.mean() - d[col].mean()) ** 2).sum() / (k - 1)
    msw = ((d[col] - g.transform("mean")) ** 2).sum() / (len(d) - k)
    return float(max((msb - msw) / (msb + (n0 - 1) * msw), 0.0)), k


def boot_corr(x, y, w, n=1000, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[ok], y[ok], w[ok]
    r = spearmanr(x, y).statistic
    uw = np.unique(w)
    idx = {u: np.nonzero(w == u)[0] for u in uw}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        k = np.concatenate([idx[u] for u in rng.choice(uw, len(uw))])
        bs.append(spearmanr(x[k], y[k]).statistic)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(r), float(lo), float(hi), int(ok.sum())


def discriminator_rf():
    # (kernel, stride) along x for TS-GAN DiscriminatorAP, 'use low' head (and 'med' head).
    trunk = [(7, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2), (3, 1), (2, 2), (3, 1)]
    low = [(3, 1), (2, 2), (3, 1), (3, 1), (2, 2), (3, 1), (1, 1)]
    med = [(3, 1)]

    def rf(layers):
        r, j = 1, 1
        for k, s in layers:
            r += (k - 1) * j
            j *= s
        return r
    return rf(trunk + med), rf(trunk + low)


def main():
    man = pd.read_csv("data/manifest.csv", dtype={"writer_id": str}).set_index("line_id")
    out = {}
    real = pd.concat([per_line_stats(pd.read_csv(f"data/dyn/words_real32_{s}.csv.gz",
                                                 dtype={"writer_id": str})) for s in ["train", "val1"]])
    real["writer"] = real.line_id.map(man.writer_id)
    real["form"] = real.line_id.map(man.form_id)
    out["variance_share"] = {c: {"writer": group_share(real, c, "writer")[0],
                                 "form": group_share(real, c, "form")[0]} for c in STATS}
    print("variance share (ICC1):", json.dumps(out["variance_share"], indent=1))

    val = real.set_index("line_id")
    lst = pd.read_csv("data/dyn/list_tsgan_val1.csv", dtype={"writer_id": str})
    style_of = dict(zip(lst.line_id, lst.style_line))
    srcs = {"real": val, "emuru": "data/dyn/words_emuru32_val1.csv.gz",
            "tsgan": "data/dyn/words_tsgan32_val1.csv.gz",
            "emuru_style": "data/dyn/fix/words_emuru_style.csv.gz",
            "tsgan_style": "data/dyn/fix/words_tsgan_style.csv.gz"}
    ok = set(pd.read_csv("data/dyn/cer_real_val1.csv").query("cer <= 0.2").line_id)
    for s in ["emuru", "tsgan"]:
        ok &= set(pd.read_csv(f"data/dyn/cer_{s}_val1.csv").query("cer <= 0.2").line_id)
    out["transfer"] = {}
    for name, src in srcs.items():
        st = src if name == "real" else per_line_stats(pd.read_csv(src, dtype={"writer_id": str})).set_index("line_id")
        ids = [l for l in st.index if l in ok and style_of.get(l) in val.index]
        w = man.loc[ids, "writer_id"].to_numpy()
        out["transfer"][name] = {}
        for c in STATS:
            x = val.loc[[style_of[l] for l in ids], c].to_numpy(float)
            y = st.loc[ids, c].to_numpy(float)
            out["transfer"][name][c] = boot_corr(x, y, w)
        print(name, {c: tuple(round(v, 2) for v in out["transfer"][name][c][:3]) for c in STATS})

    rf_med, rf_low = discriminator_rf()
    from PIL import Image
    widths = [Image.open(p).width for p in lst.image_path[:300]]
    out["tsgan_rf"] = {"rf_med_px": rf_med, "rf_low_px": rf_low, "height_px": 64,
                       "median_line_width_px": float(np.median(widths)),
                       "rf_low_share_of_line": rf_low / float(np.median(widths))}
    print(out["tsgan_rf"])
    json.dump(out, open("reports/fix/cause_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
