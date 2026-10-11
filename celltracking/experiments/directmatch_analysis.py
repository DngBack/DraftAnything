"""D1-D3 of docs/directmatch_protocol.md: LineageRAFT vs the matched-capacity direct matcher (80k steps, 3 seeds).

Per window, accuracy is averaged over the three seeds of each model; the difference LineageRAFT - DirectMatch is
summarised by its mean and a paired 95% bootstrap CI over windows, per gap, for all and dense (n_anchors > 200) windows.

  python experiments/directmatch_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402

PAIRS = {"darcy": (("raft_v3", "raft_v3_s1", "raft_v3_s2"), ("dm_darcy", "dm_darcy_s1", "dm_darcy_s2")),
         "radial": (("raft_v3_nodarcy", "raft_v3_nodarcy_s1", "raft_v3_nodarcy_s2"),
                    ("dm_radial", "dm_radial_s1", "dm_radial_s2"))}


def seed_mean(tags):
    d = pd.concat([pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet").query("method == @t + '_argmax'") for t in tags])
    return d.groupby(["window_id", "H"], as_index=False).agg(n_anchors=("n_anchors", "first"), acc=("accuracy", "mean"))


def main():
    rng = np.random.default_rng(0)
    rows = []
    for init, (raft, dm) in PAIRS.items():
        m = seed_mean(raft).merge(seed_mean(dm)[["window_id", "H", "acc"]], on=["window_id", "H"], suffixes=("_raft", "_dm"))
        for h in (33, 66, 132):
            for subset, x in (("all", m[m.H == h]), ("dense", m[(m.H == h) & (m.n_anchors > 200)])):
                d = (x.acc_raft - x.acc_dm).to_numpy()
                bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
                rows.append({"init": init, "H": h, "subset": subset, "n": len(d), "raft": x.acc_raft.mean(),
                             "dm": x.acc_dm.mean(), "diff": d.mean(), "lo": np.percentile(bs, 2.5),
                             "hi": np.percentile(bs, 97.5), "win_frac": float(np.mean(d > 0))})
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "directmatch_analysis.parquet", index=False)
    print(df.round(3).to_string(index=False))
    for name, init, subset in (("D1", "darcy", "all"), ("D2", "darcy", "dense"), ("D3", "radial", "all")):
        s = df[(df.init == init) & (df.subset == subset)]
        print(name, "holds" if (s["diff"] > 0).all() else "fails", "| CI excludes 0 at", list(s[s.lo > 0].H))


if __name__ == "__main__":
    main()
