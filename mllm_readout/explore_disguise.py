"""Probe-in-disguise analysis (exploratory, 2026-10-18). For each model/set and steering target (own = MLLM probe,
ext = SigLIP2 probe, oracle = true class, anti = wrong class, random/shuffle = R1 direction controls):
accuracy after steering, edited share, writeability W = P(steered answer = target | edited), target accuracy among
edited items, and the prediction acc_hat = acc on unedited + edited share x (W x target-right + (1 - W) x stays-right).
Output: results/explore_disguise.csv
"""
import os, numpy as np, pandas as pd
from pilot import OUT

MODELS = {m: "g3" for m in ["qwen3vl4b", "qwen3vl8b", "gemma3_4b", "gemma3_27b", "internvl35_8b", "llava15_7b", "qwen2vl_7b",
                           "qwen3vl2b", "qwen3vl32b"]} | {m: "g3_heldout" for m in ["qwen35_2b", "qwen35_4b", "qwen35_9b"]}
rows = []
for m, g3 in MODELS.items():
    for s in ["illusions", "countbench"]:
        st = np.load(f"{OUT}/{g3}/{m}_{s}_states.npz")
        y, ans = st["y"], st["ans"]
        for tgt, suf in [("own", ""), ("ext", "_ext"), ("oracle", "_oracle"), ("anti", "_anti"), ("random", "_random"), ("shuffle", "_shuffle")]:
            f = f"{OUT}/{g3}/{m}_{s}_steerA16{suf}.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f)
            e, st_, pr = d["edited"], d["steer"], d["probe"]
            W = float((st_[e] == pr[e]).mean()) if e.any() else np.nan
            rows.append(dict(model=m, set=s, target=tgt, raw=float(np.mean(ans == y)), acc=float(np.mean(st_ == y)),
                             edited=float(e.mean()), W=W, target_right=float((pr[e] == y[e]).mean()) if e.any() else np.nan,
                             probe_acc=float(np.mean(pr == y))))
D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/explore_disguise.csv", index=False)
pd.set_option("display.width", 220)
print(D.pivot_table(index=["set", "model"], columns="target", values="acc").round(3).to_string())
print("\nwriteability W = P(steered = target | edited)")
print(D.pivot_table(index=["set", "model"], columns="target", values="W").round(2).to_string())
k = D[D.target.isin(["own", "ext", "oracle", "anti"])]
print("\nacross cells and targets: corr(acc after steering, accuracy of the target) =",
      round(float(np.corrcoef(k.acc, k.probe_acc)[0, 1]), 3), f"(n={len(k)})")
print("W own vs ext (same cells):", D.pivot_table(index=["set", "model"], columns="target", values="W")[["own", "ext"]].dropna().mean().round(3).to_dict())
