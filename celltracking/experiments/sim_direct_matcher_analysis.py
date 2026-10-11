"""S1/S2 of docs/sim_direct_matcher_protocol.md: RodSim-trained direct matcher vs RodSim-trained LineageRAFT
and the ten non-learned baselines of the zero-shot table.

  python experiments/sim_direct_matcher_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402

KEY = ["dataset", "sequence", "window_id", "gap_minutes"]
BASE = ["nearest_anchor", "radial_expansion", "centroid_translation", "radial_sector",
        "ot_uot_radial", "ot_uot_none", "ot_exact_none", "cpd_radial", "darcy_68", "fsa_pos"]
DM = ["dm_sim_loc_lr", "dm_sim_loc_s1_lr", "dm_sim_loc_s2_lr"]  # amendments 1-2: the 2.5e-4 runs
DIVERGED = ["dm_sim_loc", "dm_sim_loc_s1", "dm_sim_loc_s2"]
BACTERIA = ["FAST", "recA", "rpsM", "OIAM"]


def main():
    v1 = pd.read_parquet(RESULTS / "raft_external_v1.parquet")
    v3 = pd.read_parquet(RESULTS / "raft_external_v3.parquet")
    ext = pd.read_parquet(RESULTS / "external_baselines.parquet")
    dm = pd.read_parquet(RESULTS / "raft_external_dm_sim.parquet")
    raft = pd.concat([v1[v1.method == "raft_sim_loc_argmax"],
                      v3[v3.method.isin(["raft_sim_loc_s1_argmax", "raft_sim_loc_s2_argmax"])]])
    rows = [raft.groupby(KEY, as_index=False).accuracy.mean().assign(method="raft_sim")]
    rows.append(dm[dm.method.isin([f"{t}_argmax" for t in DM])].groupby(KEY, as_index=False).accuracy.mean()
                .assign(method="dm_sim"))
    rows.append(dm[dm.method.isin([f"{t}_argmax" for t in DM + DIVERGED])])
    rows.append(v1[v1.method.isin(BASE)])
    rows.append(ext[ext.method.isin(BASE[4:])])  # geometric baselines come from v1 (also in ext)
    d = pd.concat(rows, ignore_index=True)
    # sanity: the dm windows are the zero-shot table's windows
    n_raft = raft.groupby("dataset").window_id.nunique()
    n_dm = dm.groupby("dataset").window_id.nunique()
    print("windows raft/dm:", n_raft.to_dict(), n_dm.to_dict())
    per = d.groupby(["dataset", "method", "gap_minutes"]).accuracy.mean().groupby(["dataset", "method"]).mean().unstack()
    per["best_baseline"] = per[[c for c in BASE if c in per]].max(axis=1)
    per["best_baseline_name"] = per[[c for c in BASE if c in per]].idxmax(axis=1)
    show = ["raft_sim", "dm_sim"] + [f"{t}_argmax" for t in DM + DIVERGED] + ["best_baseline", "best_baseline_name"]
    print(per[show].to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nper gap:")
    print(d[d.method.isin(["raft_sim", "dm_sim"])].groupby(["dataset", "gap_minutes", "method"]).accuracy.mean()
          .unstack().round(3).to_string())
    s1 = {k: per.loc[k, "raft_sim"] > per.loc[k, "dm_sim"] for k in BACTERIA}
    s2 = {k: per.loc[k, "dm_sim"] > per.loc[k, "best_baseline"] for k in BACTERIA}
    print("\nS1 (raft > dm on each bacterial dataset):", s1, "->", "holds" if all(s1.values()) else "fails")
    print("S2 (dm > best baseline on >=3/4):", s2, "->", "holds" if sum(s2.values()) >= 3 else "fails")
    for name in ("raft_sim", "dm_sim"):
        x = d[d.dataset.isin(BACTERIA)]
        s = x.pivot_table(index=["dataset", "sequence"], columns="method", values="accuracy")
        wins = int((s[name] > s[[c for c in BASE if c in s]].max(axis=1)).sum())
        print(f"sequence wins over all ten baselines, {name}: {wins}/{len(s)}")
    s = d[d.dataset.isin(BACTERIA)].pivot_table(index=["dataset", "sequence"], columns="method", values="accuracy")
    print("sequences where raft_sim > dm_sim:", int((s.raft_sim > s.dm_sim).sum()), "/", len(s))
    per[show].to_csv(RESULTS / "sim_direct_matcher_analysis.csv")


if __name__ == "__main__":
    main()
