"""Within-line dependence statistics for per-word residual sequences.

Lines are rows of a padded matrix R (lines x max_tokens) with NaN for invalid tokens.
Residuals are centred within each line over valid tokens, so every constant
line/writer/scan effect is removed before measuring dependence. The null permutes
valid values among valid positions of the same line (keeping the mask pattern and the
within-line marginal), optionally only within content bins.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold


# --------------------------------------------------------------------------- residuals
def make_model(kind: str):
    if kind == "ridge":
        return RidgeCV(alphas=np.logspace(-2, 3, 12))
    if kind == "gbm":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                             min_samples_leaf=40, random_state=0)
    raise ValueError(kind)


def residualize(y: np.ndarray, X: np.ndarray, is_fit: np.ndarray, groups: np.ndarray,
                kind: str, n_folds: int = 5) -> np.ndarray:
    """Content-model residuals. Rows with is_fit=True (train) are cross-fitted by group
    (form); the other rows (validation) are predicted by a model fit on all fit rows."""
    ok = np.isfinite(y)
    res = np.full(y.shape, np.nan)
    fit_idx = np.nonzero(is_fit & ok)[0]
    oth_idx = np.nonzero(~is_fit & ok)[0]
    if kind == "none":
        res[ok] = y[ok]
        return res
    for tr, te in GroupKFold(n_folds).split(fit_idx, groups=groups[fit_idx]):
        m = make_model(kind).fit(X[fit_idx[tr]], y[fit_idx[tr]])
        res[fit_idx[te]] = y[fit_idx[te]] - m.predict(X[fit_idx[te]])
    if len(oth_idx):
        m = make_model(kind).fit(X[fit_idx], y[fit_idx])
        res[oth_idx] = y[oth_idx] - m.predict(X[oth_idx])
    return res


def to_matrix(df: pd.DataFrame, col: str, pos_col: str = "pos") -> tuple[np.ndarray, np.ndarray, list]:
    """Pad per-line sequences; returns (R, line_ids order, writer per row)."""
    lines = df["line_id"].unique()
    idx = {l: i for i, l in enumerate(lines)}
    n = int(df[pos_col].max()) + 1
    R = np.full((len(lines), n), np.nan)
    R[df["line_id"].map(idx).to_numpy(), df[pos_col].to_numpy()] = df[col].to_numpy()
    writers = df.drop_duplicates("line_id").set_index("line_id").loc[lines, "writer_id"].to_numpy()
    return R, lines, writers


def detrend_lines(R: np.ndarray, x: np.ndarray | None = None, min_valid: int = 3) -> np.ndarray:
    """Remove a per-line least-squares linear trend (in x or token index), vectorised."""
    t = np.broadcast_to(np.arange(R.shape[1], dtype=float), R.shape) if x is None else x
    v = np.isfinite(R) & np.isfinite(t)
    w = v.astype(float)
    tz, yz = np.where(v, t, 0.0), np.where(v, R, 0.0)
    n, St, Sy = w.sum(1), tz.sum(1), yz.sum(1)
    Stt, Sty = (tz * tz).sum(1), (tz * yz).sum(1)
    den = n * Stt - St ** 2
    ok = (n >= min_valid) & (den > 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(ok, (n * Sty - St * Sy) / den, np.nan)
        icpt = np.where(ok, (Sy - slope * St) / n, np.nan)
    out = R - icpt[:, None] - slope[:, None] * t
    out[~v | ~ok[:, None]] = np.nan
    return out


def center_lines(R: np.ndarray, min_valid: int = 3) -> np.ndarray:
    R = R.copy()
    m = np.isfinite(R).sum(1)
    R[m < min_valid] = np.nan
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return R - np.nanmean(R, axis=1, keepdims=True)


# --------------------------------------------------------------------------- lag stats
def line_lag_terms(R: np.ndarray, lag: int):
    """Per-line numerator N, normaliser D and exact null expectation E (S=0 after centring)."""
    v = np.isfinite(R)
    Z = np.where(v, R, 0.0)
    m = v.sum(1).astype(float)
    Q = (Z ** 2).sum(1)
    S = Z.sum(1)
    pair = v[:, :-lag] & v[:, lag:]
    P = pair.sum(1).astype(float)
    N = (Z[:, :-lag] * Z[:, lag:] * pair).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(m > 0, P * Q / m, 0.0)
        E = np.where(m > 1, P * (S ** 2 - Q) / (m * (m - 1)), 0.0)
    return N, D, E


def lag_stat(R: np.ndarray, lag: int) -> tuple[float, float]:
    N, D, E = line_lag_terms(R, lag)
    return N.sum() / D.sum(), E.sum() / D.sum()


def permute_within_lines(R: np.ndarray, rng: np.random.Generator,
                         bins: np.ndarray | None = None) -> np.ndarray:
    """Permute valid values among valid positions of each line (within bins if given)."""
    v = np.isfinite(R)
    L, n = R.shape
    b = np.zeros_like(R) if bins is None else np.where(v, bins, 0.0)
    pos = np.broadcast_to(np.arange(n) / (n + 1), R.shape)
    k_dst = np.where(v, b * 2 + pos, np.inf)
    k_src = np.where(v, b * 2 + rng.random(R.shape) * 0.999, np.inf)
    dst = np.argsort(k_dst, axis=1, kind="stable")
    src = np.argsort(k_src, axis=1, kind="stable")
    rows = np.arange(L)[:, None]
    out = np.full_like(R, np.nan)
    out[rows, dst] = R[rows, src]
    out[~v] = np.nan
    return out


def perm_test(R: np.ndarray, lags, n_perm: int, seed: int,
              bins: np.ndarray | None = None, post=None) -> dict:
    """Monte Carlo within-line permutation null for several lags at once.

    post: optional transform re-applied to every permuted matrix (e.g. per-line detrend +
    centring) so the null carries the same projection bias as the observed statistic.
    Returns per lag the stats plus E_line, the per-line mean null numerator, which the
    writer bootstrap uses so the CI is relative to the same null as the test."""
    rng = np.random.default_rng(seed)
    obs = {k: line_lag_terms(R, k) for k in lags}
    nulls = {k: [] for k in lags}
    e_line = {k: np.zeros(R.shape[0]) for k in lags}
    d_line = {k: np.zeros(R.shape[0]) for k in lags}
    for _ in range(n_perm):
        P = permute_within_lines(R, rng, bins)
        if post is not None:
            P = post(P)
        for k in lags:
            N, D, _ = line_lag_terms(P, k)
            nulls[k].append(N.sum() / D.sum())
            e_line[k] += N / n_perm
            d_line[k] += D / n_perm
    out = {}
    for k in lags:
        N, D, E = obs[k]
        o, null = N.sum() / D.sum(), np.array(nulls[k])
        p = (1 + np.sum(np.abs(null - null.mean()) >= abs(o - null.mean()))) / (n_perm + 1)
        ref = e_line[k].sum() / d_line[k].sum()     # ratio-of-means null, same form as bootstrap
        out[k] = {"obs": o, "null_mean": null.mean(), "null_sd": null.std(),
                  "exact_null": E.sum() / D.sum(), "excess": o - ref,
                  "z": (o - null.mean()) / max(null.std(), 1e-12), "p": p,
                  "E_line": e_line[k], "D_line": d_line[k]}
    return out


def writer_bootstrap(R: np.ndarray, writers: np.ndarray, lag: int, n_boot: int, seed: int,
                     E: np.ndarray | None = None, Dn: np.ndarray | None = None) -> tuple[float, float]:
    """95% CI of the excess lag correlation (obs - null), resampling writers. E / Dn are the
    per-line null numerator / normaliser (from perm_test); default is the exact expectation
    of the plain within-line null, whose normaliser equals the observed one."""
    N, D, E0 = line_lag_terms(R, lag)
    E = E0 if E is None else E
    Dn = D if Dn is None else Dn
    uw, inv = np.unique(writers, return_inverse=True)
    Nw = np.bincount(inv, N, len(uw)); Dw = np.bincount(inv, D, len(uw)); Ew = np.bincount(inv, E, len(uw))
    Dnw = np.bincount(inv, Dn, len(uw))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(uw), len(uw))
        bs.append(Nw[s].sum() / Dw[s].sum() - Ew[s].sum() / Dnw[s].sum())
    return tuple(np.percentile(bs, [2.5, 97.5]))


def distance_variogram(R: np.ndarray, X: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Normalised semivariogram vs horizontal distance. Under exchangeability every pair has
    E[(r_i - r_j)^2 / 2] = Q/(m-1), so values < 1 mean nearby words are more alike."""
    num = np.zeros(len(bins) - 1); cnt = np.zeros(len(bins) - 1)
    for i in range(R.shape[0]):
        v = np.isfinite(R[i]) & np.isfinite(X[i])
        m = v.sum()
        if m < 3:
            continue
        r, x = R[i, v], X[i, v]
        q = (r ** 2).sum() / (m - 1)
        if q <= 0:
            continue
        iu = np.triu_indices(m, 1)
        g = 0.5 * (r[iu[0]] - r[iu[1]]) ** 2 / q
        d = np.abs(x[iu[0]] - x[iu[1]])
        k = np.digitize(d, bins) - 1
        ok = (k >= 0) & (k < len(bins) - 1)
        num += np.bincount(k[ok], g[ok], len(bins) - 1)
        cnt += np.bincount(k[ok], minlength=len(bins) - 1)
    with np.errstate(invalid="ignore"):
        return num / cnt, cnt
