"""Hypotheses V1-V4 of docs/sim_v3_protocol.md on raft_external_simv3.parquet.

  python experiments/raft_external_eval.py --tags <9 tags> --decoders argmax --out simv3   # GPU
  python experiments/sim_v3_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.block_bootstrap import ci  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402

ARMS = {"v3": ["raft_simv3", "raft_simv3_s1", "raft_simv3_s2"],
        "v3dm": ["dm_simv3", "dm_simv3_s1", "dm_simv3_s2"],
        "v1": ["raft_sim_loc_lr", "raft_sim_loc_s1_lr", "raft_sim_loc_s2_lr"]}
KEY = ["dataset", "sequence", "window_id", "gap_minutes"]


def arm_means():
    d = pd.read_parquet(RESULTS / "raft_external_simv3.parquet")
    out = {}
    for arm, tags in ARMS.items():
        x = d[d.method.isin([f"{t}_argmax" for t in tags])]
        assert x.method.nunique() == 3, (arm, x.method.unique())
        out[arm] = x.groupby(KEY).accuracy.mean()
    w = pd.DataFrame(out).reset_index()
    return w.merge(d[KEY + ["n_anchors"]].drop_duplicates(KEY), on=KEY, how="left", validate="one_to_one")


def oiam_ci(w, a, b, dense):
    o = w[(w.dataset == "OIAM") & ((w.n_anchors > 200) if dense else True)].copy()
    o["anchor"] = o.window_id.str.extract(r"_a(\d+)_", expand=False).astype(int)
    rows = []
    for h, x in o.sort_values(["sequence", "anchor"]).groupby("gap_minutes"):
        diffs = [(g[a] - g[b]).to_numpy() for _, g in x.groupby("sequence")]
        m, lo, hi = ci(diffs, 0)
        _, blo, bhi = ci(diffs, 4)
        rows.append({"H": h, "n": len(x), a: x[a].mean(), b: x[b].mean(), "diff": m,
                     "win_lo": lo, "win_hi": hi, "blk_lo": blo, "blk_hi": bhi})
    return pd.DataFrame(rows)


def main():
    w = arm_means()
    print("== window-mean accuracy, 3-seed mean ==")
    print(w.groupby(["dataset", "gap_minutes"])[list(ARMS)].mean().round(3).to_string())
    res = {}
    for name, a, b, dense in (("V1 v3 > v1, OIAM", "v3", "v1", False), ("V2 v3 > v1, OIAM dense", "v3", "v1", True),
                              ("V3 v3 > v3 DM, OIAM dense", "v3", "v3dm", True),
                              ("(desc) v3 > v3 DM, OIAM all", "v3", "v3dm", False)):
        r = oiam_ci(w, a, b, dense)
        res[name] = r
        print(f"\n== {name} ==\n" + r.round(3).to_string(index=False))
    v1 = res["V1 v3 > v1, OIAM"]
    v2 = res["V2 v3 > v1, OIAM dense"].query("H < 100")
    v3 = res["V3 v3 > v3 DM, OIAM dense"].query("H < 100")
    print("\nV1 holds:", bool((v1["diff"] > 0).all() and (v1.win_lo > 0).sum() >= 2))
    print("V2 holds:", bool((v2.win_lo > 0).all()))
    print("V3 holds:", bool((v3.win_lo > 0).all()))
    ext = w[w.dataset.isin(["FAST", "recA", "rpsM"])].groupby(["dataset", "gap_minutes"])[["v3", "v1"]].mean()
    ext["diff"] = ext.v3 - ext.v1
    ok = ext.groupby("dataset")["diff"].min() >= -0.02
    print("\n== V4 no harm (v3 - v1, external bacterial) ==\n" + ext.round(3).to_string())
    print("V4 holds:", bool(ok.sum() >= 2), ok.to_dict())
    big = w[(w.dataset == "OIAM") & (w.n_anchors > 400)]
    print("\n(desc) OIAM windows >400 ancestors:\n" + big.groupby("gap_minutes")[list(ARMS)].mean().round(3).to_string())


if __name__ == "__main__":
    main()
