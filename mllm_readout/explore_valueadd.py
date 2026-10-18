"""Exploratory (2026-10-18): what the LLM adds beyond its input. Grouped 5-fold probes on the SigLIP2 mean feature (ext),
the decision-token state at 75% depth (ans), and their concatenation (both, each block standardised). Value added by
the LLM = acc(both) - acc(ext); value added by the input = acc(both) - acc(ans). Output: results/explore_valueadd.csv
"""
import glob, os, re, numpy as np, pandas as pd
from pilot import probe_cv, OUT

rows = []
for p in sorted(glob.glob(f"{OUT}/g3/*_states.npz")) + sorted(glob.glob(f"{OUT}/g3_heldout/*_states.npz")):
    model, name = re.match(r"(.+)_([a-z]+)_states\.npz", os.path.basename(p)).groups()
    f = f"{OUT}/explore_input/siglip2_{name}.npz"
    if name == "mmvp" or not os.path.exists(f):
        continue
    d = np.load(p, allow_pickle=True)
    y, idx = d["y"], d["idx"]
    ok = np.isin(y, np.flatnonzero(np.bincount(y) >= 5))
    y, idx = y[ok], idx[ok]
    yy = np.unique(y, return_inverse=True)[1]
    g = np.array([re.sub(r"_px\d+", "", str(k)) for k in d["key"][ok]])
    L = round(0.75 * (d["last"].shape[1] - 1))
    z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-4)
    A = z(np.clip(np.nan_to_num(d["last"][ok][:, L + 1].astype(np.float32)), -1e4, 1e4))
    E = z(np.load(f)["feats"][idx][:, 0, 16].astype(np.float32))
    acc = {k: float((probe_cv(X, yy, groups=g)[0] == yy).mean()) for k, X in (("ext", E), ("ans", A), ("both", np.hstack([E, A])))}
    rows.append(dict(model=model, set=name, n=len(yy), **acc, llm_adds=acc["both"] - acc["ext"], input_adds=acc["both"] - acc["ans"]))
    print(f"{model:14s} {name:10s} ext {acc['ext']:.3f} ans {acc['ans']:.3f} both {acc['both']:.3f}  LLM adds {acc['both'] - acc['ext']:+.3f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/explore_valueadd.csv", index=False)
print(df.groupby("set")[["ext", "ans", "both", "llm_adds", "input_adds"]].agg(["mean", "min", "max"]).round(3).to_string())
