"""Pilot read-out (exploratory): fix rate on G3 errors, keep rate on the sampled G3-correct items, per repair and per
error type (results/g4/types_*.npz when present). Each mode is also compared with the free-generation control 'plain'
on the same items, which separates the repair from the change of read-out (decision step vs parsed generation).
"""
import glob, os, re, numpy as np, pandas as pd
from pilot import OUT

rows = []
for f in sorted(glob.glob(f"{OUT}/g4/*_pilot.npz")):
    model, name, mode = re.match(r"(.+)_([a-z]+)_([a-z0-9]+)_pilot\.npz", os.path.basename(f)).groups()
    g3 = "g3_heldout" if model.startswith(("qwen35", "qwen2vl_2b")) else "g3"
    st = np.load(f"{OUT}/{g3}/{model}_{name}_states.npz")
    wrong = dict(zip(st["idx"].tolist(), (st["ans"] != st["y"]).tolist()))
    tf = f"{OUT}/g4/types_{model}_{name}.npz"
    typ = dict(zip(np.load(tf)["idx"].tolist(), np.load(tf)["type"].tolist())) if os.path.exists(tf) else {}
    d = np.load(f)
    for i, p, y in zip(d["idx"].tolist(), d["pred"], d["y"]):
        rows.append(dict(model=model, set=name, mode=mode, idx=i, g3_wrong=wrong[i], right=bool(p == y), type=typ.get(i, "?")))
D = pd.DataFrame(rows)
if D.empty:
    raise SystemExit("no pilot files yet")
pd.set_option("display.width", 220)
s = D.groupby(["model", "set", "mode"]).apply(lambda g: pd.Series({
    "n_err": int(g.g3_wrong.sum()), "fix": g[g.g3_wrong].right.mean(), "keep": g[~g.g3_wrong].right.mean()}), include_groups=False)
print("fix = right on G3 errors; keep = right on sampled G3-correct items\n", s.round(3).to_string())
W = D.pivot_table(index=["model", "set", "idx", "g3_wrong", "type"], columns="mode", values="right").reset_index()
if "plain" in W:
    print("\nvs plain control on the same items (errors only): net = fixed_vs_plain - broken_vs_plain")
    for m in [c for c in W.columns if c in ("qfirst", "res2", "think")]:
        g = W.dropna(subset=["plain", m])
        for (model, name), h in g[g.g3_wrong].groupby(["model", "set"]):
            print(f"  {m:7s} {model:10s} {name:10s} n={len(h):3d} plain {h.plain.mean():.3f} -> {m} {h[m].mean():.3f}  "
                  f"(+{int((~h.plain.astype(bool) & h[m].astype(bool)).sum())} / -{int((h.plain.astype(bool) & ~h[m].astype(bool)).sum())})")
if (D.type != "?").any():
    print("\nfix rate per error type (pooled)\n", D[D.g3_wrong].pivot_table(index="type", columns="mode", values="right",
                                                                        aggfunc=["mean", "count"]).round(3).to_string())
