"""G5 W4 (docs/g5_protocol.md, reported, no test): steering vs directly outputting the steering target's class.

Per G3 steering run (alpha 16; targets: own decision-state probe, SigLIP2 probe, oracle, anti, random direction,
shuffled labels): accuracy of the steered answer, of the target class itself (replacing the answer by the classifier),
and of the gated replacement that uses the target only where the steering edits (target != answer).
Output: results/g5/w4.csv
"""
import glob, os, re, numpy as np, pandas as pd
from pilot import OUT

rows = []
for f in sorted(glob.glob(f"{OUT}/g3*/*_steerA16*.npz")):
    m = re.match(r"(.+)_(illusions|countbench)_steerA16_?(.*)\.npz", os.path.basename(f))
    if not m:
        continue
    model, name, tgt = m.groups()
    S = np.load(f)
    d = np.load(f"{os.path.dirname(f)}/{model}_{name}_states.npz", allow_pickle=True)
    y, ans = d["y"], d["ans"]
    rows.append(dict(model=model, set=name, target=tgt or "own", raw=np.mean(ans == y), steer=np.mean(S["steer"] == y),
                     target_acc=np.mean(S["probe"] == y), written=np.mean((S["steer"] == S["probe"])[S["probe"] != ans])))
R = pd.DataFrame(rows)
R["steer_minus_target"] = R.steer - R.target_acc
R.to_csv(f"{OUT}/g5/w4.csv", index=False)
pd.set_option("display.width", 200)
print((R.groupby(["set", "target"])[["raw", "steer", "target_acc", "steer_minus_target", "written"]].median() * 100).round(1).to_string())
print("\nn cells:", R.groupby(["set", "target"]).size().to_dict())
k = R[R.target.isin(["own", "ext"])]
print(f"steer > target class in {(k.steer_minus_target > 0.005).sum()}/{len(k)} own/ext cells; max {k.steer_minus_target.max() * 100:+.1f}")
