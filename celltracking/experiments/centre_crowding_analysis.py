"""Post hoc: margin of the moving family centre over the direct matcher against window size.

OIAM M3/M4: full model (raft_v3, 3 seeds) minus direct matcher (dm_darcy, 3 seeds), argmax, per window.
van Vliet: self-labelled adaptm minus dm_adaptm (docs/vv_family_vs_direct_protocol.md), per window.

  python experiments/centre_crowding_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402

KEY = ["sequence", "H", "window_id"]
BINS = [0, 25, 50, 100, 200, 400, 10000]


def seed_mean(tags):
    accs = []
    for t in tags:
        d = pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet")
        d = d[d.method == f"{t}_argmax"].set_index(KEY)
        accs.append(d.accuracy)
    return pd.concat(accs, axis=1).mean(axis=1), d.n_anchors


def main():
    full, n = seed_mean(["raft_v3", "raft_v3_s1", "raft_v3_s2"])
    dm, _ = seed_mean(["dm_darcy", "dm_darcy_s1", "dm_darcy_s2"])
    d = pd.DataFrame({"diff": full - dm, "n": n}).reset_index()
    d["bin"] = pd.cut(d.n, BINS)
    print("OIAM: full - direct matcher by ancestors per window")
    print(d.groupby(["H", "bin"], observed=True)["diff"].agg(["mean", "count"]).round(3).unstack("H").to_string())
    for h in (33, 66, 132):
        s = d[d.H == h]
        print(h, "Spearman(n, diff)", round(s[["n", "diff"]].corr("spearman").iloc[0, 1], 3))
    v = pd.read_parquet(RESULTS / "vv_family_vs_direct.parquet")
    w = v.pivot_table(index=["dataset", "sequence", "gap", "window_id", "n_anchors"], columns="method",
                      values="accuracy").reset_index()
    w["diff"] = w.adaptm - w.dm_adaptm
    w["bin"] = pd.cut(w.n_anchors, BINS)
    print("van Vliet: adaptm - dm_adaptm by ancestors per window")
    print(w.groupby(["gap", "bin"], observed=True)["diff"].agg(["mean", "count"]).round(3).unstack("gap").to_string())


if __name__ == "__main__":
    main()
