"""Hypotheses C1-C4 of docs/coupled_refinement_protocol.md (stage 2), OIAM M3/M4.

  python experiments/coupled_confirm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.block_bootstrap import ci  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402

ARMS = {"cpl": ["raft_cpl", "raft_cpl_s1", "raft_cpl_s2"], "v3": ["raft_v3", "raft_v3_s1", "raft_v3_s2"]}
KEY = ["sequence", "H", "window_id"]


def seed_mean(dec):
    out = {}
    for arm, tags in ARMS.items():
        ds = [pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet").query("method == @t + '_' + @dec")
              .set_index(KEY)[["accuracy", "set_f1", "n_anchors"]] for t in tags]
        assert all(len(d) for d in ds), (arm, dec)
        out[arm] = sum(d[["accuracy", "set_f1"]] for d in ds) / 3
        out["n_anchors"] = ds[0].n_anchors
    w = pd.concat({k: v for k, v in out.items() if k != "n_anchors"}, axis=1)
    w.columns = [f"{a}_{m}" for a, m in w.columns]
    return w.join(out["n_anchors"]).reset_index()


def compare(w, metric, dense):
    x = w[w.n_anchors > 200] if dense else w
    x = x.assign(anchor=x.window_id.str.extract(r"_a(\d+)_", expand=False).astype(int))
    rows = []
    for h, g in x.sort_values(["sequence", "anchor"]).groupby("H"):
        diffs = [(s[f"cpl_{metric}"] - s[f"v3_{metric}"]).to_numpy() for _, s in g.groupby("sequence")]
        m, lo, hi = ci(diffs, 0)
        _, blo, bhi = ci(diffs, 4)
        rows.append({"H": h, "n": len(g), "cpl": g[f"cpl_{metric}"].mean(), "v3": g[f"v3_{metric}"].mean(),
                     "diff": m, "win_lo": lo, "win_hi": hi, "blk_lo": blo, "blk_hi": bhi,
                     **{f"{s}_diff": float(np.mean(v)) for s, v in zip(sorted(g.sequence.unique()), diffs)}})
    return pd.DataFrame(rows)


def main():
    res = {}
    for dec in ("argmax", "mf_fam2"):
        w = seed_mean(dec)
        for metric in ("accuracy", "set_f1"):
            for dense in (False, True):
                r = compare(w, metric, dense)
                res[(dec, metric, dense)] = r
                print(f"\n== {dec} {metric} {'dense' if dense else 'all'} ==\n" + r.round(4).to_string(index=False))
    c1 = res[("argmax", "accuracy", False)]
    c2 = res[("argmax", "accuracy", True)].query("H > 40")
    c4 = res[("argmax", "set_f1", False)].query("H == 132")
    print("\nC1 holds:", bool((c1["diff"] > 0).all() and (c1.win_lo > 0).sum() >= 2))
    print("C2 holds:", bool((c2["diff"] > 0).all()))
    print("C4 holds:", bool((c4["diff"] > 0).all()))
    ce = RESULTS / "centre_error.parquet"
    if ce.exists():
        d = pd.read_parquet(ce)
        d = d[(d.n_anchors > 200) & (d.H == 132)]
        med = d.groupby("tag").err.median()
        arm = {a: float(np.mean([med[t] for t in tags])) for a, tags in ARMS.items() if all(t in med for t in tags)}
        print("\nC3 median centre error, dense 132 (spacings):", {k: round(v, 3) for k, v in arm.items()},
              "holds:", arm.get("cpl", np.inf) < arm.get("v3", -np.inf))
    pd.concat({f"{k[0]}|{k[1]}|{'dense' if k[2] else 'all'}": v for k, v in res.items()}).to_csv(
        RESULTS / "coupled_confirm.csv")


if __name__ == "__main__":
    main()
