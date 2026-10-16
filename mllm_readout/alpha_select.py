"""R2 F (docs/r2_protocol.md): steering strength selected per model on validation images, evaluated on test images.

Reads <G3>/<model>_{illusions,countbench}_steergrid.npz (steer_g3.py with STEER_GRID) for results/g3 (development
models) and results/g3_heldout. 20 random 50/50 image splits; alpha* = argmax illusions validation accuracy (ties ->
smaller alpha). Writes results/r2_alpha.csv.

usage: python alpha_select.py
"""
import glob, os
import numpy as np, pandas as pd
from pilot import OUT, probe_cv

DIRS = {"dev": f"{OUT}/g3", "heldout": f"{OUT}/g3_heldout"}
NSPLIT = 20


def calib(st):
    """cross-fitted calibrator as steer_g3.analyze"""
    y, scores, grp, key = st["y"], st["scores"], st["group"], st["key"]
    cal = np.empty_like(y)
    for g in np.unique(grp):
        k = grp == g
        cal[k] = probe_cv(scores[k], y[k], groups=key[k])[0]
    return cal


def load(d, model, name):
    st = np.load(f"{d}/{model}_{name}_states.npz")
    gr = np.load(f"{d}/{model}_{name}_steergrid.npz")
    y = st["y"]
    return dict(y=y, key=st["key"], raw=st["ans"] == y, cal=calib(st) == y, grid=gr["grid"] == y[:, None],
                alphas=gr["alphas"])


def split(key, seed):
    u, inv = np.unique(key, return_inverse=True)
    val = np.random.default_rng(seed).permutation(len(u)) < len(u) // 2
    return val[inv]


def pick(acc, val, alphas):
    m = acc[val].mean(0)
    return int(np.flatnonzero(m >= m.max() - 1e-12)[0])  # alphas ascending -> smallest best


def main():
    valid = pd.concat([pd.read_csv(f"{OUT}/g3_table.csv"), pd.read_csv(f"{OUT}/g3_heldout.csv")])
    valid = valid[(valid.n_err >= 30) & (valid.nodec <= 0.10) & (valid.text < 0.75)].cell
    rows = []
    for tag, d in DIRS.items():
        for p in sorted(glob.glob(f"{d}/*_illusions_steergrid.npz")):
            model = os.path.basename(p).replace("_illusions_steergrid.npz", "")
            if not os.path.exists(f"{d}/{model}_countbench_steergrid.npz"):
                continue
            il, cb = load(d, model, "illusions"), load(d, model, "countbench")
            A = il["alphas"]
            r = dict(model=model, set=tag, ill_valid=f"{model}_illusions" in set(valid), cb_valid=f"{model}_countbench" in set(valid))
            acc = {k: [] for k in ("il_raw", "il_cal", "il_sel", "il_a16", "cb_raw", "cb_cal", "cb_sel", "cb_own")}
            asel, aown = [], []
            for s in range(NSPLIT):
                vi, vc = split(il["key"], s), split(cb["key"], s)
                ti, tc = ~vi, ~vc
                a = pick(il["grid"], vi, A); asel.append(A[a])
                b = pick(cb["grid"], vc, A); aown.append(A[b])
                acc["il_raw"].append(il["raw"][ti].mean()); acc["il_cal"].append(il["cal"][ti].mean())
                acc["il_sel"].append(il["grid"][ti, a].mean()); acc["il_a16"].append(il["grid"][ti, list(A).index(16)].mean())
                acc["cb_raw"].append(cb["raw"][tc].mean()); acc["cb_cal"].append(cb["cal"][tc].mean())
                acc["cb_sel"].append(cb["grid"][tc, a].mean()); acc["cb_own"].append(cb["grid"][tc, b].mean())
            r.update({k: np.mean(v) for k, v in acc.items()})
            r["alpha_sel"] = float(np.median(asel)); r["alpha_sel_modal"] = float(pd.Series(asel).mode()[0])
            r["alpha_cb_own"] = float(np.median(aown))
            r["il_full"] = {f"{a:g}": round(float(il["grid"][:, i].mean()), 3) for i, a in enumerate(A)}
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/r2_alpha.csv", index=False)
    print(df.drop(columns="il_full").round(3).to_string(index=False))
    print(df[["model", "il_full"]].to_string(index=False))
    df["f1"] = df.il_sel >= np.maximum(df.il_raw, df.il_cal) + 0.10
    h = df[df.set == "heldout"]
    print(f"\nF1 held-out illusions, selected alpha >= max(raw, calib) + 10: {int(h.f1.sum())}/{len(h)} -> {'holds' if len(h) == 4 and h.f1.sum() >= 3 else 'fails/incomplete'}")
    v = df[df.ill_valid]
    print(f"F2 all valid illusion cells: {int(v.f1.sum())}/{len(v)} -> {'holds' if len(v) and v.f1.mean() >= 2/3 else 'fails'}")
    f3 = df.cb_sel < df.cb_cal
    print(f"F3 CountBench with the illusions-selected alpha below calib.: {int(f3.sum())}/{len(df)} -> {'holds' if f3.mean() >= 2/3 else 'fails'}")


if __name__ == "__main__":
    main()
