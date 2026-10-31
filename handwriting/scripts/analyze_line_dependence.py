"""H1 diagnostic (plan §4.2): do per-word geometric residuals depend on position in the line
after controlling for content, writer/line constants and scan geometry?

Main spec: ridge content model with position covariates, cross-fitted by form on train,
applied to val1; residuals centred within line; within-line permutation null; lag 1-3;
95% CI by writer bootstrap. Sensitivity: raw, GBM residuals, no-position model,
background covariate, per-line linear detrend, content-bin null.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hwline.dependence import (center_lines, detrend_lines, distance_variogram, perm_test,
                               residualize, to_matrix, writer_bootstrap)

CONTENT = ["n_chars", "n_lower", "n_upper", "n_digit", "n_punct", "n_asc", "n_desc", "n_core",
           "frac_upper", "has_desc", "has_asc_lower", "first_upper", "has_i_j"] + \
          [f"c_{c}" for c in "abcdefghijklmnopqrstuvwxyz"]
POSITION = ["rel_pos", "is_first", "is_last", "n_tokens"]
GAP_SIDE = ["n_chars", "n_upper", "n_punct", "n_asc", "n_desc", "first_upper", "n_digit"]
FEATURES = ["baseline_raw", "baseline_deskew", "log_core", "slant", "slant_grad", "log_sw", "gap", "bg"]
LAGS = [1, 2, 3]
VBINS = np.array([0, 4, 8, 12, 16, 24, 32, 48, 80], float)


def load(paths):
    df = pd.concat([pd.read_csv(p, dtype={"writer_id": str}) for p in paths], ignore_index=True)
    for c in ["v_baseline", "v_core", "v_slant", "v_slant_grad", "v_stroke"]:
        df[c] = df[c].fillna(False).astype(bool)
    med = df[df.v_core].groupby("line_id").core_h.agg(["median", "size"])
    df["med_core"] = df.line_id.map(med["median"].where(med["size"] >= 2))
    df["x_c"] = (df.ink_x0 + df.ink_x1) / 2 / df.med_core
    df["rel_pos"] = df.pos / (df.n_tokens - 1).clip(lower=1)
    df["is_first"] = (df.pos == 0).astype(float)
    df["is_last"] = (df.pos == df.n_tokens - 1).astype(float)
    df["len_bin"] = np.digitize(df.n_chars, [4, 7]).astype(float)

    b = (df.baseline_y / df.med_core).where(df.v_baseline)
    df["baseline_raw"] = b
    # Form-level scan skew: median over the form's lines of the per-line baseline slope.
    slopes = {}
    for lid, g in df[b.notna() & df.x_c.notna()].groupby("line_id"):
        if len(g) >= 3:
            slopes[lid] = np.polyfit(g.x_c, b[g.index], 1)[0]
    ls = pd.Series(slopes)
    form_slope = ls.groupby(df.drop_duplicates("line_id").set_index("line_id").loc[ls.index, "form_id"]).median()
    df["form_slope"] = df.form_id.map(form_slope)
    df["baseline_deskew"] = b - df.form_slope * df.x_c
    df["log_core"] = np.log(df.core_h).where(df.v_core)
    df["slant"] = df.slant.where(df.v_slant)
    df["slant_grad"] = df.slant_grad.where(df.v_slant_grad)
    # Words whose box overlaps a neighbour's box horizontally can share ink with it.
    df = df.sort_values(["line_id", "pos"]).reset_index(drop=True)
    g = df.groupby("line_id")
    df["overlap"] = ((df.box_x1 > g.box_x0.shift(-1)) | (df.box_x0 < g.box_x1.shift(1))).fillna(False)
    df["log_sw"] = np.log(df.stroke_w).where(df.v_stroke)
    df["bg"] = df.bg.where(df.n_ink > 0)
    alnum = df.text.astype(str).str.contains(r"[A-Za-z0-9]") & (df.n_ink > 0)
    df["alnum"] = alnum
    return df


def gap_frame(df):
    d = df.sort_values(["line_id", "pos"])
    nxt = d.groupby("line_id").shift(-1)
    ok = d.alnum & nxt.alnum.fillna(False).astype(bool) & d.med_core.notna()
    g = d.loc[ok, ["line_id", "form_id", "writer_id", "split", "pos", "n_tokens", "med_core",
                   "rel_pos", "is_first", "x_c", "len_bin", "overlap"]].copy()
    g["gap"] = ((nxt.ink_x0 - d.ink_x1) / d.med_core)[ok]
    g["is_last"] = (g.pos == g.n_tokens - 2).astype(float)
    for c in GAP_SIDE:
        g[f"l_{c}"] = d.loc[ok, c]
        g[f"r_{c}"] = nxt.loc[ok, c]
    g["bg"] = (d.bg[ok] + nxt.bg[ok]) / 2
    g["len_bin"] = np.digitize(nxt.n_chars[ok], [4, 7]).astype(float)
    return g


def residual_series(frame, feat, kind, with_pos, extra=()):
    cov = ([f"{s}_{c}" for s in "lr" for c in GAP_SIDE] if feat == "gap" else CONTENT)
    cov = cov + (POSITION if with_pos else []) + list(extra)
    X = frame[cov].fillna(0).to_numpy(float)
    y = frame[feat].to_numpy(float)
    ok_cov = np.isfinite(frame[list(extra)].to_numpy(float)).all(1) if extra else np.ones(len(frame), bool)
    y = np.where(ok_cov, y, np.nan)
    return residualize(y, X, (frame.split == "train").to_numpy(), frame.form_id.to_numpy(), kind)


def analyse(frame, res, split, detrend, n_perm, n_boot, seed, bins=False):
    sub = frame.assign(res=res)[frame.split == split]
    sub = sub[np.isfinite(sub.res)]
    R, lines, writers = to_matrix(sub, "res")
    post = None
    if detrend:
        X, *_ = to_matrix(sub, "x_c")
        post = lambda M: center_lines(detrend_lines(M, X), min_valid=4)
        R = post(R)
    else:
        R = center_lines(R, min_valid=3)
    B = None
    if bins:
        B, *_ = to_matrix(sub, "len_bin")
    tests = perm_test(R, LAGS, n_perm, seed, B, post)
    out = []
    for lag in LAGS:
        t = tests[lag]
        E, Dn = t.pop("E_line"), t.pop("D_line")
        t["ci_lo"], t["ci_hi"] = writer_bootstrap(R, writers, lag, n_boot, seed + lag, E, Dn)
        t["lag"] = lag
        t["n_lines"] = int((np.isfinite(R).sum(1) >= 3).sum())
        t["n_writers"] = len(np.unique(writers[np.isfinite(R).sum(1) >= 3]))
        t["n_values"] = int(np.isfinite(R).sum())
        out.append(t)
    return out, R, writers, sub, lines


def stability(R, writers, seed):
    """Lag-1 excess by number of valid words per line, and on two random writer halves."""
    from hwline.dependence import line_lag_terms
    N, D, E = line_lag_terms(R, 1)
    m = np.isfinite(R).sum(1)
    out = {}
    for name, sel in [("3-5", (m >= 3) & (m <= 5)), ("6-8", (m >= 6) & (m <= 8)), ("9+", m >= 9)]:
        out[f"len_{name}"] = float((N[sel] - E[sel]).sum() / max(D[sel].sum(), 1e-12))
    uw = np.unique(writers)
    rng = np.random.default_rng(seed)
    half = set(rng.permutation(uw)[: len(uw) // 2])
    h = np.array([w in half for w in writers])
    for name, sel in [("writers_A", h), ("writers_B", ~h)]:
        out[name] = float((N[sel] - E[sel]).sum() / max(D[sel].sum(), 1e-12))
    # share of writers (>=3 usable lines) whose own lag-1 excess is positive
    pos, tot = 0, 0
    for w in uw:
        s = (writers == w) & (m >= 3)
        if s.sum() >= 3 and D[s].sum() > 0:
            tot += 1
            pos += (N[s] - E[s]).sum() > 0
    out["frac_writers_pos"] = pos / max(tot, 1)
    out["n_writers_3lines"] = tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/h1"))
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    words = load(["data/words_train.csv.gz", "data/words_val1.csv.gz"])
    gaps = gap_frame(words)

    specs = {  # name -> (kind, with_pos, extra covariates, detrend, content-bin null, drop overlapping)
        "main_ridge_pos": ("ridge", True, (), False, False, False),
        "raw": ("none", False, (), False, False, False),
        "gbm_pos": ("gbm", True, (), False, False, False),
        "ridge_nopos": ("ridge", False, (), False, False, False),
        "ridge_pos_bg": ("ridge", True, ("bg",), False, False, False),
        "ridge_pos_detrend": ("ridge", True, (), True, False, False),
        "ridge_pos_binnull": ("ridge", True, (), False, True, False),
        "ridge_pos_noovl": ("ridge", True, (), False, False, True),
    }
    rows, stab, vario = [], [], {}
    for feat in FEATURES:
        frame = gaps if feat == "gap" else words
        for sname, (kind, wpos, extra, detr, bins, drop_ovl) in specs.items():
            if feat == "bg" and extra:
                continue
            res = residual_series(frame, feat, kind, wpos, extra)
            if drop_ovl:
                res = np.where(frame.overlap.to_numpy(bool), np.nan, res)
            for split in ["train", "val1"]:
                n_perm = a.n_perm if sname == "main_ridge_pos" else a.n_perm // 2
                out, R, writers, sub, lines = analyse(frame, res, split, detr, n_perm, a.n_boot, a.seed, bins)
                for t in out:
                    rows.append({"feature": feat, "spec": sname, "split": split, **t})
                if sname == "main_ridge_pos":
                    stab.append({"feature": feat, "split": split, **stability(R, writers, a.seed)})
                    X, *_ = to_matrix(sub, "x_c")
                    vario[(feat, split)] = distance_variogram(R, X, VBINS)
            print(feat, sname, "done", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(a.out / "lag_stats.csv", index=False)
    pd.DataFrame(stab).to_csv(a.out / "stability.csv", index=False)
    vt = pd.DataFrame([{"feature": f, "split": s, "d_lo": VBINS[i], "d_hi": VBINS[i + 1],
                        "gamma_norm": g[0][i], "n_pairs": g[1][i]}
                       for (f, s), g in vario.items() for i in range(len(VBINS) - 1)])
    vt.to_csv(a.out / "variogram.csv", index=False)
    plot(tab, vt, a.out)
    decide(tab, a.out)


def plot(tab, vt, out):
    feats = FEATURES
    fig, axes = plt.subplots(1, len(feats), figsize=(3.0 * len(feats), 3.2), sharey=True)
    for ax, f in zip(axes, feats):
        for split, col, dx in [("train", "#2a6fdb", -0.08), ("val1", "#e07b39", 0.08)]:
            t = tab[(tab.feature == f) & (tab.spec == "main_ridge_pos") & (tab.split == split)]
            ax.errorbar(t.lag + dx, t.excess, yerr=[t.excess - t.ci_lo, t.ci_hi - t.excess],
                        fmt="o", color=col, label=split, capsize=3)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f, fontsize=10)
        ax.set_xticks(LAGS)
        ax.set_xlabel("lag (words)")
    axes[0].set_ylabel("lag corr. excess over\nwithin-line permutation null")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "lag_excess.png", dpi=130)

    fig, axes = plt.subplots(1, len(feats), figsize=(3.0 * len(feats), 3.2), sharey=True)
    for ax, f in zip(axes, feats):
        for split, col in [("train", "#2a6fdb"), ("val1", "#e07b39")]:
            t = vt[(vt.feature == f) & (vt.split == split)]
            ax.plot((t.d_lo + t.d_hi) / 2, t.gamma_norm, "o-", color=col, label=split, ms=3)
        ax.axhline(1, color="0.5", lw=0.8)
        ax.set_title(f, fontsize=10)
        ax.set_xlabel("distance (core heights)")
    axes[0].set_ylabel("normalised semivariogram\n(1 = exchangeable)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "variogram.png", dpi=130)


def decide(tab, out):
    """Pre-stated internal rule (plan §4.2): lag-1 excess CI > 0 in train AND val1 under the
    main spec, and the sign survives GBM residuals, content-bin null and background covariate."""
    l1 = tab[tab.lag == 1].set_index(["feature", "spec", "split"])
    verdict = {}
    for f in FEATURES:
        def pos(spec, split):
            k = (f, spec, split)
            return bool(k in l1.index and l1.loc[k, "ci_lo"] > 0)
        main = pos("main_ridge_pos", "train") and pos("main_ridge_pos", "val1")
        ctrls = {s: pos(s, "val1") for s in ["gbm_pos", "ridge_pos_binnull", "ridge_pos_bg",
                                             "ridge_pos_noovl", "ridge_pos_detrend"]
                 if (f, s, "val1") in l1.index}
        verdict[f] = {"main_replicates": main, "controls_val1_ci_pos": ctrls,
                      "excess_train": float(l1.loc[(f, "main_ridge_pos", "train"), "excess"]),
                      "excess_val1": float(l1.loc[(f, "main_ridge_pos", "val1"), "excess"])}
    (out / "decision.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
