"""Analysis for docs/matched_trackastra_protocol.md (M1-M3) and docs/pseudo_direct_matcher_protocol.md (P1-P2).

  python experiments/matched_trackastra_analysis.py matched
  python experiments/matched_trackastra_analysis.py pseudo
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.decoder_on_baselines import paired_ci  # noqa: E402
from experiments.family_set import RESULTS, ROOT  # noqa: E402

KEY = ["sequence", "H", "window_id"]
HS = (33, 66, 132)
CKPT = ROOT / "outputs/checkpoints/trackastra_ft"


def seed_mean(tags, name):
    """Per-window accuracy averaged over seeds (argmax), with the dense flag."""
    parts = []
    for t in tags:
        d = pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet")
        parts.append(d[d.method == f"{t}_argmax"].set_index(KEY)[["accuracy", "n_anchors"]])
    acc = pd.concat([p.accuracy for p in parts], axis=1)
    out = pd.DataFrame({name: acc.mean(1), "n_anchors": parts[0].n_anchors})
    if acc.isna().any().any():
        print(f"warning: {name} seeds cover different windows", flush=True)
    return out, acc


def trackastra(fname, name):
    d = pd.read_parquet(RESULTS / f"{fname}.parquet")
    return d[d.method == "trackastra_ft_argmax"].set_index(KEY).accuracy.rename(name)


def m2_val(suffix, h):
    hist = pd.read_csv(CKPT / f"h{h}{suffix}" / "history.csv")
    col = f"val_acc_{h}" if f"val_acc_{h}" in hist else "val_acc"
    hist = hist[hist.step > 0]
    return float(hist[col].max()), int(hist.loc[hist[col].idxmax(), "step"]), int(hist.step.max())


def report(df, cols, ref):
    for h in HS:
        s = df.xs(h, level="H")
        line = " ".join(f"{c} {s[c].mean():.3f}" for c in cols)
        dense = s[s.n_anchors > 200]
        dline = " ".join(f"{c} {dense[c].mean():.3f}" for c in cols)
        print(f"H={h} (n={len(s)}): {line} | dense (n={len(dense)}): {dline}")
        for c in cols:
            if c != ref:
                m, lo, hi = paired_ci(s[ref], s[c])
                print(f"   {ref} - {c}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def matched():
    m01, per_seed = seed_mean(["raft_m01_20k", "raft_m01_20k_s1", "raft_m01_20k_s2"], "matched")
    full, _ = seed_mean(["raft_v3", "raft_v3_s1", "raft_v3_s2"], "full")
    df = m01.join(full.full).join(trackastra("trackastra_ft", "ta_gap")).join(trackastra("trackastra_ft_multi", "ta_multi"))
    assert not df.isna().any().any(), df[df.isna().any(axis=1)]
    print("per-seed matched:\n", per_seed.groupby(level="H").mean().round(3).to_string())
    report(df, ["matched", "ta_gap", "ta_multi", "full"], "matched")
    print("M2 validation (best acc, step, last step):")
    for h in HS:
        g, m = m2_val("", h), m2_val("_multi", h)
        print(f"  H={h}: per-gap {g}, multi {m} -> selected {'multi' if m[0] > g[0] else 'per-gap'}")
    for h in HS:
        s = df.xs(h, level="H")
        print(f"  multi - per-gap H={h}: " + "{:+.3f} [{:+.3f}, {:+.3f}]".format(*paired_ci(s.ta_multi, s.ta_gap)))


def pseudo():
    dmp, per_seed = seed_mean(["dm_pseudo", "dm_pseudo_s1", "dm_pseudo_s2"], "dm_pseudo")
    v3p, _ = seed_mean(["raft_v3p", "raft_v3p_s1", "raft_v3p_s2"], "raft_pseudo")
    dm, _ = seed_mean(["dm_darcy", "dm_darcy_s1", "dm_darcy_s2"], "dm_true")
    df = v3p.join(dmp.dm_pseudo).join(dm.dm_true)
    assert not df.isna().any().any()
    print("per-seed dm_pseudo:\n", per_seed.groupby(level="H").mean().round(3).to_string())
    report(df, ["raft_pseudo", "dm_pseudo", "dm_true"], "raft_pseudo")
    for h in HS:
        s = df.xs(h, level="H")
        print(f"  dm_true - dm_pseudo H={h}: {s.dm_true.mean() - s.dm_pseudo.mean():+.3f}")


if __name__ == "__main__":
    {"matched": matched, "pseudo": pseudo}[sys.argv[1]]()
