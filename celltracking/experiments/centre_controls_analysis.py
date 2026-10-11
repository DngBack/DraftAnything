"""K1-K3 of docs/centre_controls_protocol.md: LineageRAFT against three controls for the moving family centre.

Per window, accuracy is averaged over the three seeds of each model; the difference LineageRAFT - control is
summarised by its mean and a paired 95% bootstrap CI over windows, per gap, for all and dense (n_anchors > 200) windows.

  python experiments/centre_controls_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.directmatch_analysis import seed_mean  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402

RAFT = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
ARMS = {arm: (arm, f"{arm}_s1", f"{arm}_s2") for arm in ("dm_k96", "dm_aux", "raft_T1")}


def main():
    rng = np.random.default_rng(0)
    ref = seed_mean(RAFT)
    rows = []
    for arm, tags in ARMS.items():
        if not all((RESULTS / f"raft_eval_{t}.parquet").exists() for t in tags):
            print(arm, "incomplete, skipped")
            continue
        m = ref.merge(seed_mean(tags)[["window_id", "H", "acc"]], on=["window_id", "H"], suffixes=("_raft", "_arm"))
        for h in (33, 66, 132):
            for subset, x in (("all", m[m.H == h]), ("dense", m[(m.H == h) & (m.n_anchors > 200)])):
                d = (x.acc_raft - x.acc_arm).to_numpy()
                bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
                rows.append({"arm": arm, "H": h, "subset": subset, "n": len(d), "raft": x.acc_raft.mean(),
                             "arm_acc": x.acc_arm.mean(), "diff": d.mean(), "lo": np.percentile(bs, 2.5),
                             "hi": np.percentile(bs, 97.5), "win_frac": float(np.mean(d > 0))})
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "centre_controls_analysis.parquet", index=False)
    print(df.round(3).to_string(index=False))
    for name, arm, gaps in (("K1", "dm_k96", (33, 66)), ("K2", "dm_aux", (33, 66)), ("K3", "raft_T1", (33,))):
        s = df[(df.arm == arm) & (df.subset == "all") & df.H.isin(gaps)] if len(df) else df
        if len(s):
            print(name, "holds" if (s.lo > 0).all() else "fails", "| CI excludes 0 at", list(s[s.lo > 0].H))
    for t in sum(ARMS.values(), ()):
        log = Path(__file__).parents[1] / "outputs" / "logs" / f"{t}.log"
        rec = [ln for ln in log.read_text().replace("\r", "\n").splitlines() if "recall=" in ln] if log.exists() else []
        if rec:
            print(t, "final training", rec[-1].split("recall=")[1].split()[0], "recall")


if __name__ == "__main__":
    main()
