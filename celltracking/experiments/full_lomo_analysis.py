"""L1-L3 of docs/full_lomo_protocol.md: leave-one-movie-out for the full model.

  python experiments/full_lomo_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402

FOLDS = ("M0", "M1", "M2", "M3", "M4")
H = (33, 66, 132)


def acc(tag):
    d = pd.read_parquet(RESULTS / f"raft_eval_{tag}.parquet").query("method == @tag + '_argmax'")
    d["dense"] = d.n_anchors > 200
    return d


def main():
    old = pd.read_parquet(RESULTS / "leave_one_movie_multihorizon_all.parquet")
    rows = []
    for k in FOLDS:
        full, norot = acc(f"raft_v3_lomo_{k}"), acc(f"raft_lomo_{k}")
        o = old[(old.method == "multi_horizon") & (old.held_out_movie == k)].copy()
        o["acc"] = o.n_correct / o.n_scored
        for h in H:
            f, n = full[full.H == h], norot[norot.H == h]
            rows.append({"fold": k, "H": h, "full": f.accuracy.mean(), "norot30k": n.accuracy.mean(),
                         "gbdt": o[o.horizon_frames == h].acc.mean(), "full_dense": f[f.dense].accuracy.mean(),
                         "norot_dense": n[n.dense].accuracy.mean(), "n_windows": len(f), "n_dense": int(f.dense.sum())})
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "full_lomo_analysis.parquet", index=False)
    print(df.round(3).to_string(index=False))
    print(df.groupby("H")[["full", "norot30k", "gbdt", "full_dense", "norot_dense"]].mean().round(3).to_string())
    l1 = sum(all(df[(df.fold == k) & (df.H == h)].eval("full > norot30k").item() for h in (33, 66)) for k in FOLDS)
    l2 = sum(all(df[(df.fold == k) & (df.H == h)].eval("full > gbdt").item() for h in H) for k in FOLDS)
    m33 = df[df.H == 33].full.mean()
    print(f"L1 {'holds' if l1 >= 4 else 'fails'} ({l1}/5 folds beat no-rot 30k at 33 and 66)")
    print(f"L2 {'holds' if l2 >= 4 else 'fails'} ({l2}/5 folds beat GBDT at every gap)")
    print(f"L3 {'holds' if m33 >= 0.90 else 'fails'} (mean at 33 = {m33:.3f})")


if __name__ == "__main__":
    main()
