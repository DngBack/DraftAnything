"""G2 decision (docs/g2_protocol.md): per-error decomposition + gated repair vs calib-only / steer-all.

gated (label-free): if probe agrees with calibration -> calib answer; else -> steered answer (toward probe class).
Needs repair parquets with probe_pred/new columns (G2 runs onward).
"""
import glob, os, sys
import numpy as np, pandas as pd
from pilot import probe_cv, OUT
from analyze_g15 import cell


def gated(tag):
    d = np.load(f"{OUT}/{tag}_states.npz")
    y, ans = d["y"], d["ans"]
    cal = probe_cv(d["scores"], y)[0]
    r = pd.read_parquet(f"{OUT}/{tag}_repair.parquet")
    r = r[r.primary].set_index("item").sort_index()
    out = np.where(r.probe_pred.values == cal, cal, r.new.values)
    ok = ans == y
    return dict(steer_all=r.now_correct.mean(), gated=(out == y).mean(),
                gated_break=1 - (out == y)[ok].mean(), steer_break=1 - r.now_correct.values[ok].mean())


def main(fams=("attr", "spatial_real")):
    rows = []
    for p in sorted(glob.glob(f"{OUT}/*_states.npz")):
        tag = os.path.basename(p).replace("_states.npz", "")
        if not tag.endswith(tuple(fams)):
            continue
        rows.append({**cell(p), **gated(tag)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/g2_table.csv", index=False)
    print(df.drop(columns=["probe_at", "cal_recovers"]).round(3).to_string(index=False))
    ok = df[df.n_err >= 30]
    win = (ok.gated >= ok[["calib", "steer_all"]].max(axis=1)).mean()
    brk = (ok.gated_break <= 0.02).all()
    print(f"\ncells >=30 err: {len(ok)} | gated >= max(calib, steer-all): {win:.2f} | all gated break <= 2%: {brk} | "
          f"verdict: {'GO (repair headline)' if win >= 2 / 3 and brk else 'diagnostic-only'}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("attr", "spatial_real", "color_cf"))
