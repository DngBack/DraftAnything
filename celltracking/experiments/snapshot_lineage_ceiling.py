"""Ceiling probe: how much would knowing the target-frame family partition help?

Development analysis on OIAM M3/M4 (already viewed). Families come from GT and
are an *oracle*; the point is to bound the value of predicting a hierarchical
grouping ("genealogy from a snapshot") before building a predictor.

  python experiments/snapshot_lineage_ceiling.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.pairwise_geometry_baseline import HORIZONS  # noqa: E402


def geometric_costs(e):
    src = e["anchor"][["center_y", "center_x"]].to_numpy(dtype=float)
    dst = e["target"][["center_y", "center_x"]].to_numpy(dtype=float)
    vs, vt = src - src.mean(0), dst - dst.mean(0)
    scale = np.sqrt(len(dst) / len(src))
    return {"nearest": ((dst[:, None] - src[None]) ** 2).sum(-1),
            "radial": ((vt[:, None] - scale * vs[None]) ** 2).sum(-1)}


def cluster_match(cost, groups):
    """Assign every group to one distinct anchor (Hungarian on summed cost)."""
    labels = np.unique(groups)
    block = np.stack([cost[groups == g].sum(0) for g in labels])
    big = block.max() * 10 + 1
    n_g, n_a = block.shape
    if n_g > n_a:  # more groups than anchors: allow reuse via duplicated anchors
        reps = int(np.ceil(n_g / n_a))
        block = np.concatenate([block + k * big for k in range(reps)], axis=1)
    rows, cols = linear_sum_assignment(block)
    out = np.empty(len(groups), dtype=int)
    for r, c in zip(rows, cols):
        out[groups == labels[r]] = c % n_a
    return out


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        for h in HORIZONS:
            for e in examples(alias, windows, compensate=False, horizon=h):
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                tid = e["target"].local_detection_id.to_numpy()
                aid = e["anchor"].local_detection_id.to_numpy()
                row_of = {int(v): i for i, v in enumerate(tid)}
                col_of = {int(v): i for i, v in enumerate(aid)}
                ok = ok[ok.target_detection_id.isin(row_of) & ok.ancestor_detection_id.isin(col_of)]
                if not len(ok):
                    continue
                r = ok.target_detection_id.map(row_of).to_numpy()
                truth = ok.ancestor_detection_id.map(col_of).to_numpy()
                for name, cost in geometric_costs(e).items():
                    c = cost[r]
                    preds = {f"{name}_per_target": c.argmin(1),
                             f"{name}_oracle_partition": cluster_match(c, truth)}
                    for m, p in preds.items():
                        rows.append({"sequence": alias, "H": h,
                                     "window_id": e["window"]["window_id"],
                                     "n_anchors": len(aid), "method": m,
                                     "accuracy": float(np.mean(p == truth))})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "snapshot_lineage_ceiling.parquet", index=False)
    df["dense"] = df.n_anchors > 200
    print(df.groupby(["H", "method"]).accuracy.mean().unstack().round(4).to_string())
    print(df[df.dense].groupby(["H", "method"]).accuracy.mean().unstack().round(4).to_string())


if __name__ == "__main__":
    main()
