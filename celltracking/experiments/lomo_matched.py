"""Hypotheses M1-M4 of docs/lomo_matched_protocol.md: full (raft_v3_lomo_*), direct matcher (dm_lomo_*) and coupled
(cpl_lomo_*) leave-one-movie-out models, each evaluated on its held-out movie.

  python experiments/lomo_matched.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402

ARMS = {"full": "raft_v3_lomo", "dm": "dm_lomo", "cpl": "cpl_lomo"}
MOVIES = ("M0", "M1", "M2", "M3", "M4")
# amendment 2026-10-12: dm_lomo_M3 diverged (CE > 10 from step 25k) -> rerun once at lr 2.5e-4, used in M1/M2;
# the diverged run is reported as arm "dm_div"
RERUN = {("dm", "M3"): "dm_lomo_M3_lr"}
EXTRA = {"dm_div": ("dm_lomo", "M3")}


def frame():
    out = {}
    for arm, pre in ARMS.items():
        ds = []
        for m in MOVIES:
            t = RERUN.get((arm, m), f"{pre}_{m}")
            d = pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet").query("method == @t + '_argmax'")
            ds.append(d.set_index(["sequence", "H", "window_id"])[["accuracy", "n_anchors"]])
        out[arm] = pd.concat(ds)
    w = pd.DataFrame({a: v.accuracy for a, v in out.items()})
    w["n_anchors"] = out["full"].n_anchors
    for arm, (pre, m) in EXTRA.items():
        t = f"{pre}_{m}"
        d = pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet").query("method == @t + '_argmax'")
        w[arm] = d.set_index(["sequence", "H", "window_id"]).accuracy
    assert not w[list(ARMS)].isna().any().any(), "window sets differ between arms"
    return w.reset_index()


def strat_ci(x, a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [(g[a] - g[b]).to_numpy() for _, g in x.groupby("sequence")]
    allv = np.concatenate(groups)
    s = [np.concatenate([v[rng.integers(0, len(v), len(v))] for v in groups]).mean() for _ in range(n)]
    return allv.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    w = frame()
    w["dense"] = w.n_anchors > 200
    print("== per fold (all windows) ==")
    cols = list(ARMS) + list(EXTRA)
    print(w.groupby(["sequence", "H"])[cols].mean().round(3).unstack("H").to_string())
    print("== per fold (dense) ==")
    print(w[w.dense].groupby(["sequence", "H"])[cols].mean().round(3).unstack("H").to_string())
    folds = w.groupby(["sequence", "H"])[list(ARMS)].mean()
    wins = (folds.full > folds.dm).groupby("H").sum()
    print("M1 folds full > dm:", wins.to_dict(), "holds:", bool(wins[33] >= 4 and wins[66] >= 4))
    rows = []
    for (name, a, b, dense) in (("full-dm", "full", "dm", False), ("full-dm", "full", "dm", True),
                                ("cpl-full", "cpl", "full", False), ("cpl-full", "cpl", "full", True)):
        for h, g in (w[w.dense] if dense else w).groupby("H"):
            m, lo, hi = strat_ci(g, a, b)
            rows.append({"cmp": name, "sub": "dense" if dense else "all", "H": h, "n": len(g), "diff": m, "lo": lo, "hi": hi})
    r = pd.DataFrame(rows)
    print(r.round(4).to_string(index=False))
    r.to_csv(RESULTS / "lomo_matched.csv", index=False)
    q = r.set_index(["cmp", "sub", "H"])
    print("M2 holds:", bool(all(q.loc[("full-dm", "dense", h), "lo"] > 0 for h in (33, 66))))
    print("M3 holds:", bool(all(q.loc[("cpl-full", "dense", h), "diff"] > 0 for h in (66, 132))))
    print("M4 holds:", bool(all(q.loc[("cpl-full", "all", h), "diff"] >= -0.01 for h in (33, 66, 132))))


if __name__ == "__main__":
    main()
