"""R1-R3 of docs/trackastra_norot_protocol.md: Trackastra fine-tuned without rotation augmentation.

  python experiments/trackastra_norot_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.directmatch_analysis import seed_mean  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402

SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
H = (33, 66, 132)


def per_window(name, method):
    d = pd.read_parquet(RESULTS / f"{name}.parquet").query("method == @method")
    return d[["window_id", "H", "n_anchors", "n_scored", "accuracy"]].rename(columns={"accuracy": "acc"})


def paired(a, b, rng):
    m = a.merge(b[["window_id", "H", "acc"]], on=["window_id", "H"], suffixes=("_a", "_b"))
    rows = []
    for h in H:
        for subset, x in (("all", m[m.H == h]), ("dense", m[(m.H == h) & (m.n_anchors > 200)])):
            d = (x.acc_a - x.acc_b).to_numpy()
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
            rows.append({"H": h, "subset": subset, "n": len(d), "a": x.acc_a.mean(), "b": x.acc_b.mean(),
                         "diff": d.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
    return pd.DataFrame(rows)


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "_norot"  # _norot (batch 16) or _norot_b8 (amendment)
    rng = np.random.default_rng(0)
    off = per_window("trackastra_ft", "trackastra_ft_argmax")
    nor = per_window(f"trackastra_ft{suffix}", "trackastra_ft_argmax")
    print("Trackastra FT (all / dense), argmax and greedy")
    for name in ("trackastra_ft", f"trackastra_ft{suffix}"):
        d = pd.read_parquet(RESULTS / f"{name}.parquet")
        d["dense"] = d.n_anchors > 200
        print(name, "\n", d.pivot_table(index="method", columns="H", values="accuracy").round(3).to_string())
        print(" dense\n", d[d.dense].pivot_table(index="method", columns="H", values="accuracy").round(3).to_string())
        w = d[d.method == "trackastra_ft_argmax"]
        print(" cell-weighted", [round((w[w.H == h].accuracy * w[w.H == h].n_scored).sum() / w[w.H == h].n_scored.sum(), 3) for h in H])
    r1 = paired(nor, off, rng)
    print("\nR1 norot - official\n", r1.round(3).to_string(index=False))
    n_better = int((r1[r1.subset == "all"]["diff"] > 0).sum())
    print("R1", "holds" if n_better >= 2 else "fails", f"({n_better}/3 gaps improved)")
    # better Trackastra row chosen on M2 validation accuracy (checkpoint files), per gap
    import torch
    ck = Path(__file__).parents[1] / "outputs" / "checkpoints" / "trackastra_ft"
    best = []
    for h in H:
        v = {s: torch.load(ck / f"h{h}{s}" / "best.pt", map_location="cpu", weights_only=False)["val_acc"] for s in ("", suffix)}
        s = max(v, key=v.get)
        print(f"H{h} M2 val: official {v['']:.4f} {suffix} {v[suffix]:.4f} -> {s or 'official'}")
        best.append((nor if s else off).query("H == @h"))
    best = pd.concat(best)
    raft = seed_mean(SEEDS)
    r2 = paired(raft, best, rng)
    print("\nR2 full LineageRAFT (3-seed mean) - better Trackastra\n", r2.round(3).to_string(index=False))
    s = r2[(r2.subset == "all") & r2.H.isin([33, 66])]
    print("R2", "holds" if (s.lo > 0).all() else "fails")
    ens = pd.read_parquet(RESULTS / f"image_partition_decode_{'+'.join(SEEDS)}.parquet")
    ens = ens.rename(columns={"mf_img2": "acc"})[["window_id", "H", "n_anchors", "acc"]]
    r3 = paired(ens, best, rng)
    print("\nR3 decoded ensemble - better Trackastra\n", r3.round(3).to_string(index=False))
    print("R3", "holds" if (r3[r3.subset == "all"]["diff"] > 0).all() else "fails")
    pd.concat([r1.assign(test="R1"), r2.assign(test="R2"), r3.assign(test="R3")]).to_parquet(
        RESULTS / f"trackastra{suffix}_analysis.parquet", index=False)


if __name__ == "__main__":
    main()
