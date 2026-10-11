"""Non-rigid Coherent Point Drift (Myronenko & Song, TPAMI 2010) baseline for two-snapshot ancestry.

Ancestors are the GMM centroids, moved by a Gaussian-kernel displacement field (width beta,
regularisation lam) onto the target snapshot; each target goes to the centroid with the largest
posterior. A natural fit: each ancestor is one Gaussian that explains several descendants.
Coordinates are in units of the anchor spacing; optional radial pre-alignment as in ot_baseline.

Same protocol as ot_baseline: grid chosen on OIAM M0-M2 (mean over 33/66/132), evaluated once on M3/M4.

  python experiments/cpd_baseline.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.ot_baseline import align, windows  # noqa: E402

GRID = [{"beta": b, "lam": l, "w": w} for b, l, w in itertools.product((2.0, 4.0, 8.0, 16.0, 32.0), (1.0, 4.0, 16.0), (0.0, 0.1))]
ALIGN = ("none", "radial")


def cpd(src, dst, beta, lam, w, iters=150, tol=1e-6):
    """Return the posterior argmax (target -> ancestor) after non-rigid CPD of src onto dst."""
    s = max(float(np.median(cKDTree(src).query(src, k=2)[0][:, 1])), 1.0)
    Y = torch.from_numpy(src / s).double()
    X = torch.from_numpy(dst / s).double()
    M, N, D = len(Y), len(X), 2
    G = torch.exp(-torch.cdist(Y, Y) ** 2 / (2 * beta ** 2))
    W = torch.zeros(M, D, dtype=torch.float64)
    T = Y.clone()
    sigma2 = float((torch.cdist(X, Y) ** 2).sum()) / (D * M * N)
    for _ in range(iters):
        d2 = torch.cdist(T, X) ** 2  # M x N
        P = torch.exp(-d2 / (2 * sigma2))
        c = (2 * np.pi * sigma2) ** (D / 2) * w / (1 - w) * M / N if w > 0 else 0.0
        P = P / (P.sum(0, keepdim=True) + c + 1e-300)
        P1, Pt1, PX = P.sum(1), P.sum(0), P @ X
        A = P1[:, None] * G + lam * sigma2 * torch.eye(M, dtype=torch.float64)
        W = torch.linalg.solve(A, PX - P1[:, None] * Y)
        T = Y + G @ W
        Np = float(P1.sum())
        new = float((Pt1[:, None] * X * X).sum() - 2 * (PX * T).sum() + (P1[:, None] * T * T).sum()) / (Np * D)
        new = max(new, 1e-6)
        if abs(sigma2 - new) < tol:
            sigma2 = new
            break
        sigma2 = new
    return torch.cdist(X, T).argmin(1).numpy()


def one(wd, configs):
    torch.set_num_threads(2)
    alias, h, wid, shape, src, dst, r, truth = wd
    moved = {a: align(src, dst, h, shape, a) for a in ALIGN}
    rows = []
    for name, (a, p) in configs.items():
        pred = cpd(moved[a], dst, **p)[r]
        rows.append({"sequence": alias, "H": h, "window_id": wid, "n_anchors": len(src),
                     "n_scored": len(r), "method": name, "accuracy": float(np.mean(pred == truth))})
    print(alias, h, wid, flush=True)
    return rows


def run(aliases, configs, jobs=16):
    ws = sorted(windows(aliases), key=lambda w: -len(w[5]))
    out = Parallel(n_jobs=jobs)(delayed(one)(w, configs) for w in ws)
    return pd.DataFrame([r for rows in out for r in rows])


def label(a, p):
    return f"cpd_{a}_" + "_".join(f"{k}{v}" for k, v in p.items())


def main():
    configs = {label(a, p): (a, p) for a in ALIGN for p in GRID}
    dev = run(["M0", "M1", "M2"], configs)
    dev.to_parquet(RESULTS / "cpd_baseline_dev.parquet", index=False)
    score = dev.groupby(["method", "H"]).accuracy.mean().groupby("method").mean()
    best = {}
    for a in ALIGN:
        names = [n for n in score.index if n.startswith(f"cpd_{a}_")]
        best[f"cpd_{a}"] = configs[score[names].idxmax()]
        print("dev best", f"cpd_{a}", score[names].idxmax(), round(score[names].max(), 4), flush=True)
    test = run(["M3", "M4"], best)
    test.to_parquet(RESULTS / "cpd_baseline.parquet", index=False)
    t = test.assign(dense=test.n_anchors > 200)
    print(t.groupby(["method", "H"]).accuracy.mean().unstack().round(3).to_string())
    print("dense\n", t[t.dense].groupby(["method", "H"]).accuracy.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
