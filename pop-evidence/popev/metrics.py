"""Bag-level metrics with combination-level bootstrap intervals.

Labels are constant within a combination, so the unit of replication is the combination (or
"partition:combination" when pooling partitions), never the image or the bag. Primary endpoint
(protocol.md §5): macro AP over species with both positive and negative combinations.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    _, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    last = np.cumsum(cnt).astype(float)
    ranks = np.empty(len(s)); ranks[order] = ((last - cnt + 1 + last) / 2)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(s: np.ndarray, y: np.ndarray) -> float:
    """Step-wise AP (sklearn's definition); ties broken pessimistically are not modelled."""
    y = y.astype(bool)
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    tp = np.cumsum(y[o]); prec = tp / np.arange(1, len(y) + 1)
    return float(prec[y[o]].sum() / npos)


def argmax_f1_thresholds(S: np.ndarray, Y: np.ndarray, n_grid: int = 199) -> np.ndarray:
    thr = np.zeros(S.shape[1])
    for k in range(S.shape[1]):
        grid = np.quantile(S[:, k], np.linspace(0, 1, n_grid + 1)[:-1])
        y = Y[:, k].astype(bool)
        best = -1.0
        for t in grid:
            p = S[:, k] > t
            tp = (p & y).sum(); f = 2 * tp / max(p.sum() + y.sum(), 1)
            if f > best:
                best, thr[k] = f, t
    return thr


class _Prepared:
    """Sort orders and tie groups computed once per (seed, species), so that a combination
    bootstrap is a re-weighting of bags (weight = times its combination was drawn) and every
    resample costs O(M) per species instead of a sort. With unit weights the results equal
    `auroc` and `average_precision` exactly; with weights, AP treats a duplicated bag as one
    bag of weight w (precision is not stepped within the duplicates)."""

    def __init__(self, S: np.ndarray, Y: np.ndarray, P: np.ndarray) -> None:
        self.S, self.Y, self.P = S, Y.astype(bool), P.astype(bool)
        R, M, K = S.shape
        self.desc = np.empty((R, K, M), int); self.asc = np.empty((R, K, M), int)
        self.groups = [[None] * K for _ in range(R)]
        for r in range(R):
            for k in range(K):
                self.desc[r, k] = np.argsort(-S[r, :, k], kind="mergesort")
                o = np.argsort(S[r, :, k], kind="mergesort"); self.asc[r, k] = o
                s = S[r, o, k]
                self.groups[r][k] = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]

    def point(self, r: int, w: np.ndarray) -> Dict:
        Y, P = self.Y, self.P[r]; K = Y.shape[1]; W = w.sum()
        wm = lambda v: float((w * v).sum() / W)
        tp = (Y & P).sum(1); d = 2 * tp + (P & ~Y).sum(1) + (Y & ~P).sum(1)
        f1 = wm(np.where(d > 0, 2 * tp / np.maximum(d, 1), 1.0))
        f1_const = wm(2 * Y.sum(1) / (Y.sum(1) + K))
        ap, au, mf1, fpr, rec = [], [], [], [], []
        for k in range(K):
            y = Y[:, k]; wp, wn = (w * y).sum(), (w * ~y).sum()
            if wp == 0 or wn == 0:
                ap.append(np.nan); au.append(np.nan)
            else:
                o = self.desc[r, k]; wo, yo = w[o], y[o]
                prec = np.cumsum(wo * yo) / np.maximum(np.cumsum(wo), 1e-12)
                ap.append(float((wo * yo * prec).sum() / wp))
                o = self.asc[r, k]; wo, yo = w[o], y[o]; g = self.groups[r][k]
                pg = np.add.reduceat(wo * yo, g); ng = np.add.reduceat(wo * ~yo, g)
                below = np.cumsum(ng) - ng
                au.append(float((pg * (below + 0.5 * ng)).sum() / (wp * wn)))
            p = P[:, k]
            t = (w * (y & p)).sum(); fp = (w * (~y & p)).sum(); fn = (w * (y & ~p)).sum()
            mf1.append(2 * t / max(2 * t + fp + fn, 1e-12) if (2 * t + fp + fn) > 0 else 0.0)
            fpr.append(fp / wn if wn > 0 else np.nan); rec.append(t / wp if wp > 0 else np.nan)
        return {"macro_ap": float(np.nanmean(ap)), "auroc_mean": float(np.nanmean(au)),
                "f1": f1, "f1_margin": f1 - f1_const, "macro_f1": float(np.mean(mf1)),
                "exact": wm((Y == P).all(1)), "n_pred": wm(P.sum(1)), "n_true": wm(Y.sum(1)),
                "fpr_mean": float(np.nanmean(fpr)), "ap": ap, "auroc": au, "fpr": fpr, "recall": rec}

    def seeds(self, w: np.ndarray) -> Dict:
        pts = [self.point(r, w) for r in range(self.S.shape[0])]
        out = {k: float(np.mean([q[k] for q in pts])) for k in SCALARS}
        for k in ("ap", "auroc", "fpr", "recall"):
            out[k] = list(np.nanmean([q[k] for q in pts], axis=0))
        return out


def _point(S, Y, P) -> Dict:
    return _Prepared(np.asarray(S)[None], Y, np.asarray(P)[None]).point(0, np.ones(len(Y)))


SCALARS = ("macro_ap", "auroc_mean", "f1", "f1_margin", "macro_f1", "exact", "n_pred", "fpr_mean")


def _as3(a):
    return np.asarray(a)[None] if np.ndim(a) == 2 else np.asarray(a)


def _boot_weights(units, n_boot: int, seed: int):
    """Yields (M,) bag weights: how many times each bag's unit was drawn in one resample."""
    U = np.asarray(units); uniq, inv = np.unique(U, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        yield cnt[inv].astype(float)


def report(S: np.ndarray, Y: np.ndarray, P: np.ndarray, units: Sequence[str], class_names: List[str],
           n_boot: int = 2000, seed: int = 0) -> Dict:
    """S scores, P binary predictions (thresholds may differ per fold), (M, K) or (R seeds, M, K);
    Y labels (M, K). Point estimates and intervals are seed-averaged; resampling unit = `units`."""
    prep = _Prepared(_as3(S), Y, _as3(P))
    pt = prep.seeds(np.ones(len(Y)))
    boots = {k: [] for k in SCALARS}; per = {k: [[] for _ in class_names] for k in ("ap", "auroc")}
    for w in _boot_weights(units, n_boot, seed):
        b = prep.seeds(w)
        for k in SCALARS:
            boots[k].append(b[k])
        for m in per:
            for j, v in enumerate(b[m]):
                if v == v:
                    per[m][j].append(v)
    ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])] if len(v) else [np.nan, np.nan]
    out = {k: {"point": pt[k], "ci95": ci(boots[k])} for k in SCALARS}
    U = np.asarray(units); uniq = sorted(set(U)); first = {u: np.flatnonzero(U == u)[0] for u in uniq}
    Yb = Y.astype(bool)
    out["per_species"] = {
        n: {"ap": pt["ap"][j], "ap_ci95": ci(per["ap"][j]), "auroc": pt["auroc"][j], "auroc_ci95": ci(per["auroc"][j]),
            "fpr": pt["fpr"][j], "recall": pt["recall"][j],
            "n_pos_combos": sum(Yb[first[u], j] for u in uniq), "n_neg_combos": sum(not Yb[first[u], j] for u in uniq)}
        for j, n in enumerate(class_names)}
    out["n_units"] = len(uniq)
    return out


def paired_delta(A: tuple, B: tuple, Y: np.ndarray, units: Sequence[str], keys=("macro_ap", "auroc_mean", "fpr_mean"),
                 n_boot: int = 2000, seed: int = 0) -> Dict:
    """Metric(A) - metric(B) on the same bags, with the same combination resamples for both.
    A, B = (S, P), each (M, K) or (R, M, K). Returns point, 95 % interval, share of resamples > 0."""
    pa, pb = _Prepared(_as3(A[0]), Y, _as3(A[1])), _Prepared(_as3(B[0]), Y, _as3(B[1]))
    one = np.ones(len(Y)); a, b = pa.seeds(one), pb.seeds(one)
    d = {k: [] for k in keys}
    for w in _boot_weights(units, n_boot, seed):
        ra, rb = pa.seeds(w), pb.seeds(w)
        for k in keys:
            d[k].append(ra[k] - rb[k])
    return {k: {"point": a[k] - b[k], "ci95": [float(x) for x in np.nanpercentile(d[k], [2.5, 97.5])],
                "p_gt0": float(np.mean(np.asarray(d[k]) > 0))} for k in keys}


def context_sensitivity(S: np.ndarray, Y: np.ndarray, combos: Sequence[str], class_names: List[str]) -> Dict:
    """Per species: spread of the per-combination mean score over positive combinations (S_k), the
    same over negatives, the positive-negative gap and the combination-level AUROC. S_k alone is not
    a quality measure: a constant predictor has S_k = 0."""
    C = np.asarray(combos); uniq = sorted(set(C))
    m = np.stack([S[C == c].mean(0) for c in uniq]); y = np.stack([Y[C == c][0] for c in uniq]).astype(bool)
    out = {}
    for j, n in enumerate(class_names):
        p, q = m[y[:, j], j], m[~y[:, j], j]
        out[n] = {"S_pos": float(p.std()) if len(p) else np.nan, "S_neg": float(q.std()) if len(q) else np.nan,
                  "gap": float(p.mean() - q.mean()) if len(p) and len(q) else np.nan,
                  "combo_auroc": auroc(m[:, j], y[:, j])}
    return out


def by_combination(S: np.ndarray, Y: np.ndarray, P: np.ndarray, combos: Sequence[str], class_names: List[str]) -> List[Dict]:
    C = np.asarray(combos); rows = []
    for c in sorted(set(C), key=lambda c: (c.count("_"), c)):
        ii = C == c; y = Y[ii][0].astype(bool); p = P[ii].astype(bool)
        tp = (p & y).sum(1); f1 = (2 * tp / np.maximum(p.sum(1) + y.sum(), 1)).mean()
        rows.append({"combo": c, "order": int(y.sum()), "n_bags": int(ii.sum()), "f1": float(f1),
                     "exact": float((p == y).all(1).mean()),
                     "pred_rate": {n: float(p[:, j].mean()) for j, n in enumerate(class_names)},
                     "mean_score": {n: float(S[ii, j].mean()) for j, n in enumerate(class_names)}})
    return rows


def fmt(d: Dict) -> str:
    return f"{d['point']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def _attr_fit(m: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.c_[np.ones(len(y)), y]
    return np.linalg.lstsq(X, np.log(m / (1 - m)), rcond=None)[0][1:].T


def context_attribution(S: np.ndarray, Y: np.ndarray, combos: Sequence[str], n_boot: int = 0, seed: int = 0) -> Dict:
    """Own/context decomposition over combinations: least squares of logit(mean score of k | c) on the
    presence vector y_c of all species. B[k, k] = own evidence, B[k, j != k] = context. With n_boot > 0,
    95 % intervals from resampling combinations; `inverted_resolved` counts species whose own-evidence
    interval lies below 0. Descriptive: with C combinations each row has K + 1 parameters."""
    C = np.asarray(combos); uniq = sorted(set(C))
    m = np.stack([S[C == c].mean(0) for c in uniq]).clip(1e-4, 1 - 1e-4)
    y = np.stack([Y[C == c][0] for c in uniq]).astype(float)
    B = _attr_fit(m, y)
    off = ~np.eye(len(B), dtype=bool)
    out = {"B": B.tolist(), "own": np.diag(B).tolist(), "own_mean": float(np.diag(B).mean()),
           "context_mean_abs": float(np.abs(B[off]).mean()), "n_inverted": int((np.diag(B) < 0).sum())}
    if n_boot:
        rng = np.random.default_rng(seed); own, ctx = [], []
        for _ in range(n_boot):
            ii = rng.integers(0, len(uniq), len(uniq))
            if np.linalg.matrix_rank(np.c_[np.ones(len(ii)), y[ii]]) < y.shape[1] + 1:
                continue
            b = _attr_fit(m[ii], y[ii]); own.append(np.diag(b)); ctx.append(np.abs(b[off]).mean())
        if not own:                       # every resample rank-deficient (too few units): no intervals
            nan = [[np.nan, np.nan]] * len(B)
            out.update({"own_ci95": nan, "context_ci95": [np.nan, np.nan], "inverted_resolved": 0,
                        "positive_resolved": 0, "n_boot_used": 0})
            return out
        own = np.asarray(own); lo, hi = np.percentile(own, [2.5, 97.5], axis=0)
        out.update({"own_ci95": np.c_[lo, hi].tolist(), "context_ci95": np.percentile(ctx, [2.5, 97.5]).tolist(),
                    "inverted_resolved": int((hi < 0).sum()), "positive_resolved": int((lo > 0).sum()),
                    "n_boot_used": len(own)})
    return out


def decorrelation_weights(Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Importance weights that turn the empirical label-vector distribution into the product of its
    marginals: w(y) = prod_k p(y_k) / p_hat(y). Simulates a test set with the same class frequencies but
    no co-occurrence structure (a combination-shift stress test for in-distribution datasets).
    alpha < 1 tempers the weights; label vectors are counted with add-alpha smoothing."""
    Y = np.asarray(Y).astype(int)
    keys, inv, cnt = np.unique(Y, axis=0, return_inverse=True, return_counts=True)
    p_emp = (cnt[inv] + alpha) / (len(Y) + alpha * len(keys))
    m = Y.mean(0)
    p_ind = np.prod(np.where(Y == 1, m, 1 - m), axis=1)
    w = p_ind / p_emp
    return w / w.mean()


def decorrelated(S: np.ndarray, Y: np.ndarray, P: np.ndarray) -> Dict:
    """Seed-averaged metrics under decorrelation weights."""
    prep = _Prepared(_as3(S), Y, _as3(P))
    return prep.seeds(decorrelation_weights(Y))


def rank_within(S: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per split (fold or partition) and species, replace scores by their rank fraction among that split's
    bags. Removes score-scale differences between separately trained models before pooling; uses no labels."""
    from scipy.stats import rankdata
    S3 = np.asarray(S)[None] if np.ndim(S) == 2 else np.asarray(S)
    out = np.zeros_like(S3, dtype=float); g = np.asarray(groups)
    for u in np.unique(g):
        ii = g == u
        for r in range(S3.shape[0]):
            for k in range(S3.shape[2]):
                out[r, ii, k] = rankdata(S3[r, ii, k]) / ii.sum()
    return out if np.ndim(S) == 3 else out[0]
