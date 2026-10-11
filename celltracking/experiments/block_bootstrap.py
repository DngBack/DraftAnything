"""Time-block bootstrap for the main OIAM M3/M4 comparisons (docs/block_bootstrap_protocol.md).

  python experiments/block_bootstrap.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402

HS = (33, 66, 132)
SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
DM = ("dm_darcy", "dm_darcy_s1", "dm_darcy_s2")
B = 5000


def seed_mean(tags):
    d = pd.concat([pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet").query("method == @t + '_argmax'") for t in tags])
    return d.groupby(["H", "window_id"]).accuracy.mean()


def per_window():
    """Two frames of paired per-window accuracies: all windows, and the decoder windows."""
    full = seed_mean(SEEDS).rename("full")
    dm = seed_mean(DM).rename("dm")
    tr = pd.read_parquet(RESULTS / "trackastra_ft.parquet").query("method == 'trackastra_ft_argmax'")
    tr = tr.set_index(["H", "window_id"]).accuracy.rename("trackastra")
    g = pd.read_parquet(RESULTS / "leave_one_movie_multihorizon_all.parquet")
    g = g[(g.method == "multi_horizon") & g.held_out_movie.isin(["M3", "M4"])]
    g = (g.n_correct / g.n_scored).groupby([g.horizon_frames.rename("H"), g.window_id]).mean().rename("gbdt")
    allw = pd.concat([full, dm, tr, g], axis=1).dropna()
    d = pd.read_parquet(RESULTS / "decoder_on_baselines.parquet")
    dec = d.pivot_table(index=["H", "window_id"], columns="method", values="accuracy")
    with open(RESULTS / "partition_metrics_cache.pkl", "rb") as f:
        pm = {(it["H"], it["window_id"]): float(np.mean(it["pred"]["ens_noann_mf2"] == it["truth"])) for it in pickle.load(f)}
    dec["noann"] = [pm[k] for k in dec.index]
    return allw, dec.dropna()


def blocks(n, L, rng):
    if n <= L:
        return rng.integers(0, n, n)
    starts = rng.integers(0, n - L + 1, int(np.ceil(n / L)))
    return (starts[:, None] + np.arange(L)[None]).ravel()[:n]


def ci(diff_by_movie, L, seed=0):
    """diff_by_movie: list of arrays ordered by anchor frame. L = 0: window bootstrap (pooled)."""
    rng = np.random.default_rng(seed)
    allv = np.concatenate(diff_by_movie)
    stats = np.empty(B)
    for b in range(B):
        if L == 0:
            stats[b] = allv[rng.integers(0, len(allv), len(allv))].mean()
        else:
            stats[b] = np.concatenate([v[blocks(len(v), L, rng)] for v in diff_by_movie]).mean()
    return allv.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def run(frame, name, a, b):
    rows = []
    for h in HS:
        x = frame.loc[h].copy()
        x["seq"] = x.index.str[:2]
        x["anchor"] = x.index.str.extract(r"_a(\d+)_", expand=False).astype(int)
        x = x.sort_values(["seq", "anchor"])
        dm = [(g[a] - g[b]).to_numpy() for _, g in x.groupby("seq")]
        r = {"comparison": name, "H": h, "n": len(x)}
        for L, lab in ((0, "window"), (4, "block4"), (8, "block8")):
            m, lo, hi = ci(dm, L)
            r.update({"mean": m, f"{lab}_lo": lo, f"{lab}_hi": hi})
        r.update({f"{s}_diff": float(np.mean(v)) for s, v in zip(sorted(x.seq.unique()), dm)})
        rows.append(r)
    return rows


def main():
    allw, dec = per_window()
    rows = []
    rows += run(allw, "C1 full - Trackastra FT", "full", "trackastra")
    rows += run(allw, "C2 full - direct matcher", "full", "dm")
    rows += run(allw, "C3 full - GBDT", "full", "gbdt")
    rows += run(dec, "C4 ens+dec - Trackastra FT", "raft_ens_mf2", "trackastra_argmax")
    rows += run(dec, "C5 noann ens+dec - Trackastra FT", "noann", "trackastra_argmax")
    rows += run(dec, "C6a full+dec - Trackastra+dec (fitted T)", "raft_mf2", "trackastra_mf2")
    rows += run(dec, "C6b full+dec - Trackastra+dec (T=1)", "raft_mf2", "trackastra_mf2_T1")
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "block_bootstrap.parquet", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    held = (df.window_lo > 0)
    print("B1: comparisons with window CI > 0:", int(held.sum()), "of which block4 CI > 0:", int((held & (df.block4_lo > 0)).sum()))
    print(df[held & ~(df.block4_lo > 0)][["comparison", "H"]].to_string(index=False))


if __name__ == "__main__":
    main()
