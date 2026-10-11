"""Optimal-transport baselines for two-snapshot ancestry (Waddington-OT style).

Anchors carry mass r = N_B/N_A (uniform growth), targets mass 1. Couplings:
  exact : capacity-constrained min-cost assignment (each anchor takes at most
          ceil(c * r) targets), solved exactly by linear assignment;
  uot   : entropic OT with a hard target marginal and a KL-relaxed source
          marginal (unknown per-ancestor growth), target -> argmax of the plan.
Costs are squared distances in units of the anchor spacing, after an optional
pre-alignment of the anchors: none, radial expansion about the centroid by
sqrt(r), or Darcy advection (darcy_flow_prior.advect).

Hyperparameters are chosen on OIAM M0-M2 (mean over 33/66/132 min) and the
chosen setting per variant is evaluated once on M3/M4, the windows of tab:oiam.

  python experiments/ot_baseline.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import torch
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.darcy_flow_prior import advect  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.lineage_glue import image_shape_of  # noqa: E402

HS = (33, 66, 132)
GRID = {"exact": [{"c": c} for c in (1.0, 1.25, 1.5, 2.0, 3.0)],
        "uot": [{"eps": e, "rho": r} for e, r in itertools.product((0.05, 0.2, 0.5, 1.0, 2.0), (0.1, 1.0, 10.0))]}
ALIGN = ("none", "radial", "darcy")


def align(src, dst, h, shape, how):
    if how == "radial":
        c = src.mean(0)
        return dst.mean(0) + np.sqrt(len(dst) / len(src)) * (src - c)
    if how == "darcy":
        return advect(src, dst, h, shape)
    return src


def couple(src, dst, kind, p):
    s = max(float(np.median(cKDTree(src).query(src, k=2)[0][:, 1])), 1.0)
    C = ((dst[:, None, :] - src[None, :, :]) ** 2).sum(-1) / s ** 2  # targets x anchors
    r = len(dst) / len(src)
    if kind == "exact":
        k = max(int(np.ceil(p["c"] * r)), int(np.ceil(len(dst) / len(src))))
        row, col = linear_sum_assignment(np.tile(C, (1, k)))
        out = np.empty(len(dst), int)
        out[row] = col % len(src)
        return out
    eps, rho = p["eps"], p["rho"]
    Ct = torch.from_numpy(C).float()
    loga = float(np.log(r))
    f, g = torch.zeros(len(src)), torch.zeros(len(dst))
    lam = rho / (rho + eps)
    for _ in range(300):
        g = -eps * torch.logsumexp((f[None, :] - Ct) / eps, dim=1)
        f = lam * eps * (loga - torch.logsumexp((g[:, None] - Ct) / eps, dim=0))
    return (f[None, :] - Ct).argmax(1).numpy()


def windows(aliases):
    w = Windows(BENCH, ROOT / "data").windows
    for alias in aliases:
        shape = image_shape_of(alias)
        for h in HS:
            for e in examples(alias, w, compensate=False, horizon=h):
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok) or len(co) < 2:
                    continue
                yield (alias, h, e["window"]["window_id"], shape,
                       e["anchor"][["center_y", "center_x"]].to_numpy(float),
                       e["target"][["center_y", "center_x"]].to_numpy(float),
                       ok.target_detection_id.map(ro).to_numpy(), ok.ancestor_detection_id.map(co).to_numpy())


def one(w, configs):
    torch.set_num_threads(2)
    alias, h, wid, shape, src, dst, r, truth = w
    moved = {a: align(src, dst, h, shape, a) for a in ALIGN}
    rows = []
    for name, (a, kind, p) in configs.items():
        pred = couple(moved[a], dst, kind, p)[r]
        rows.append({"sequence": alias, "H": h, "window_id": wid, "n_anchors": len(src),
                     "n_scored": len(r), "method": name, "accuracy": float(np.mean(pred == truth))})
    print(alias, h, wid, flush=True)
    return rows


def run(aliases, configs):
    ws = sorted(windows(aliases), key=lambda w: -len(w[5]))  # largest first for load balance
    out = Parallel(n_jobs=16)(delayed(one)(w, configs) for w in ws)
    return pd.DataFrame([r for rows in out for r in rows])


def label(a, kind, p):
    return f"{kind}_{a}_" + "_".join(f"{k}{v}" for k, v in p.items())


def main():
    configs = {label(a, k, p): (a, k, p) for a in ALIGN for k, ps in GRID.items() for p in ps}
    dev = run(["M0", "M1", "M2"], configs)
    dev.to_parquet(RESULTS / "ot_baseline_dev.parquet", index=False)
    score = dev.groupby(["method", "H"]).accuracy.mean().groupby("method").mean()
    best = {}
    for a in ALIGN:
        for k in GRID:
            names = [n for n in score.index if n.startswith(f"{k}_{a}_")]
            best[f"ot_{k}_{a}"] = configs[score[names].idxmax()]
            print("dev best", f"ot_{k}_{a}", score[names].idxmax(), round(score[names].max(), 4))
    test = run(["M3", "M4"], best)
    test.to_parquet(RESULTS / "ot_baseline.parquet", index=False)
    t = test.assign(dense=test.n_anchors > 200)
    print(t.groupby(["method", "H"]).accuracy.mean().unstack().round(3).to_string())
    print("dense\n", t[t.dense].groupby(["method", "H"]).accuracy.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
