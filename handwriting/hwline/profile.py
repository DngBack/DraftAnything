"""Source-agnostic within-line dynamics profile (real lines or any generator's output).

measure_line: segment a line image into its written words and compute per-word descriptors.
dynamics_profile: per feature, the statistics used to compare sources:
  lag1_excess      lag-1 residual correlation over the within-line permutation null
  lag1_detrended   same after removing a per-line linear trend in x (null re-detrends)
  trend_r2_excess  share of within-line variance explained by a linear trend in x, minus its
                   permutation-null expectation (how much of the variation is drift)
  within_sd        median within-line SD of residuals (how much words vary at all)
Residuals come from a ridge content model cross-fitted by line within each source, so
every source is judged relative to its own content effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from hwline.dependence import (center_lines, detrend_lines, perm_test, to_matrix,
                               writer_bootstrap)
from hwline.descriptors import content_features, slant_gradient, word_descriptors
from hwline.segment import segment_words, written_tokens

FEATURES = ["baseline", "gap", "slant_grad", "log_core", "log_sw"]
CONTENT = ["n_chars", "n_lower", "n_upper", "n_digit", "n_punct", "n_asc", "n_desc", "n_core",
           "frac_upper", "has_desc", "has_asc_lower", "first_upper", "has_i_j"] + \
          [f"c_{c}" for c in "abcdefghijklmnopqrstuvwxyz"] + ["rel_pos", "is_first", "is_last"]


def measure_line(gray: np.ndarray, transcript: str, meta: dict, min_margin: float = 0.0):
    toks = written_tokens(transcript)
    boxes, thr, margin = segment_words(gray, len(toks))
    if boxes is None or margin < min_margin:
        return [], False, margin
    rows = []
    for i, (t, b) in enumerate(zip(toks, boxes)):
        d = word_descriptors(gray, b, thr, t)
        x0, y0, x1, y1 = b
        sg = slant_gradient(gray[y0:y1, x0:x1], gray[y0:y1, x0:x1] < thr) \
            if sum(c.isalnum() for c in t) >= 2 else None
        d["slant_grad"], d["v_slant_grad"] = (sg, True) if sg is not None else (np.nan, False)
        rows.append({**meta, "pos": i, "n_tokens": len(toks), "text": t, "box_x0": x0,
                     "box_y0": y0, "box_x1": x1, "box_y1": y1, "line_h": gray.shape[0],
                     **d, **content_features(t)})
    return rows, True, margin


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["line_id", "pos"]).reset_index(drop=True)
    for c in ["v_baseline", "v_core", "v_slant_grad", "v_stroke"]:
        df[c] = df[c].fillna(False).astype(bool)
    med = df[df.v_core].groupby("line_id").core_h.agg(["median", "size"])
    df["med_core"] = df.line_id.map(med["median"].where(med["size"] >= 2))
    df["x_c"] = (df.ink_x0 + df.ink_x1) / 2 / df.med_core
    df["rel_pos"] = df.pos / (df.n_tokens - 1).clip(lower=1)
    df["is_first"] = (df.pos == 0).astype(float)
    df["is_last"] = (df.pos == df.n_tokens - 1).astype(float)
    df["baseline"] = (df.baseline_y / df.med_core).where(df.v_baseline)
    df["log_core"] = np.log(df.core_h).where(df.v_core)
    df["slant_grad"] = df.slant_grad.where(df.v_slant_grad)
    df["log_sw"] = np.log(df.stroke_w).where(df.v_stroke)
    nxt = df.groupby("line_id").shift(-1)
    alnum = df.text.astype(str).str.contains(r"[A-Za-z0-9]") & (df.n_ink > 0)
    ok = alnum & nxt.text.astype(str).str.contains(r"[A-Za-z0-9]").fillna(False) & df.med_core.notna()
    df["gap"] = ((nxt.ink_x0 - df.ink_x1) / df.med_core).where(ok)
    return df


def _residuals(df: pd.DataFrame, feat: str) -> np.ndarray:
    y = df[feat].to_numpy(float)
    X = df[CONTENT].fillna(0).to_numpy(float)
    ok = np.isfinite(y)
    res = np.full(len(df), np.nan)
    idx = np.nonzero(ok)[0]
    if len(idx) < 50:
        return res
    for tr, te in GroupKFold(5).split(idx, groups=df.line_id.to_numpy()[idx]):
        m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[idx[tr]], y[idx[tr]])
        res[idx[te]] = y[idx[te]] - m.predict(X[idx[te]])
    return res


def trend_r2(R: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-line share of (centred) variance explained by a linear fit in x."""
    D = detrend_lines(R, X, min_valid=4)
    v = np.isfinite(D)
    Rc = center_lines(np.where(v, R, np.nan), min_valid=4)
    tot = np.nansum(Rc ** 2, 1)
    res = np.nansum(D ** 2, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((v.sum(1) >= 4) & (tot > 0), 1 - res / tot, np.nan)


def dynamics_profile(df: pd.DataFrame, n_perm: int = 200, n_boot: int = 500, seed: int = 0,
                     features=FEATURES) -> list[dict]:
    df = add_features(df)
    out = []
    for feat in features:
        res = _residuals(df, feat)
        sub = df.assign(res=res)
        sub = sub[np.isfinite(sub.res) & np.isfinite(sub.x_c)]
        if sub.line_id.nunique() < 30:
            continue
        R0, lines, writers = to_matrix(sub, "res")
        X, *_ = to_matrix(sub, "x_c")
        R = center_lines(R0, 3)
        row = {"feature": feat, "n_lines": int((np.isfinite(R).sum(1) >= 3).sum()),
               "n_values": int(np.isfinite(R).sum())}
        t = perm_test(R, [1], n_perm, seed)[1]
        row["lag1_excess"] = t["excess"]
        row["lag1_lo"], row["lag1_hi"] = writer_bootstrap(R, writers, 1, n_boot, seed,
                                                          t["E_line"], t["D_line"])
        post = lambda M: center_lines(detrend_lines(M, X), min_valid=4)
        Rd = post(R0)
        td = perm_test(Rd, [1], n_perm, seed, post=post)[1]
        row["lag1_detrended"] = td["excess"]
        row["lag1d_lo"], row["lag1d_hi"] = writer_bootstrap(Rd, writers, 1, n_boot, seed,
                                                            td["E_line"], td["D_line"])
        # Trend share vs the within-line permutation null, bootstrap over writers.
        rng = np.random.default_rng(seed)
        r2 = trend_r2(R, X)
        from hwline.dependence import permute_within_lines
        null = np.nanmean([trend_r2(permute_within_lines(R, rng), X) for _ in range(n_perm // 4)], 0)
        ex = r2 - null
        okl = np.isfinite(ex)
        row["trend_r2_excess"] = float(np.nanmean(ex))
        uw, inv = np.unique(writers[okl], return_inverse=True)
        s_w = np.bincount(inv, ex[okl], len(uw)); n_w = np.bincount(inv, minlength=len(uw))
        bs = []
        for _ in range(n_boot):
            k = rng.integers(0, len(uw), len(uw))
            bs.append(s_w[k].sum() / n_w[k].sum())
        row["trend_lo"], row["trend_hi"] = np.percentile(bs, [2.5, 97.5])
        row["within_sd"] = float(np.nanmedian(np.nanstd(R, 1)[np.isfinite(R).sum(1) >= 3]))
        out.append(row)
    return out


def per_line_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per line: baseline slope (core h per 10 core h of x), linear-fit R2, baseline SD, gap SD/CV."""
    f = add_features(df.copy())
    rows = []
    for lid, g in f.groupby("line_id"):
        b = g[np.isfinite(g.baseline) & np.isfinite(g.x_c)]
        gp = g.gap.dropna()
        slope, r2 = np.nan, np.nan
        if len(b) >= 4:
            k, c = np.polyfit(b.x_c, b.baseline, 1)
            ss = ((b.baseline - b.baseline.mean()) ** 2).sum()
            slope = k
            r2 = 1 - ((b.baseline - (k * b.x_c + c)) ** 2).sum() / ss if ss > 0 else np.nan
        rows.append({"line_id": lid, "slope": slope * 10, "r2": r2,  # slope: core h per 10 core h of x
                     "base_sd": b.baseline.std() if len(b) >= 4 else np.nan,
                     "gap_sd": gp.std() if len(gp) >= 3 else np.nan,
                     "gap_cv": gp.std() / gp.mean() if len(gp) >= 3 and gp.mean() > 0 else np.nan})
    return pd.DataFrame(rows)
