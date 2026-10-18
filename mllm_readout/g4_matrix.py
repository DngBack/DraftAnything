"""G4 repair matrix and predictions P1-P6 (docs/g4_protocol.md).

Rows: error types from explore_types.py (results/g4/types_<model>_<set>.npz). Columns: R1 calibration (cross-fitted
calibrator), R2 label-free hidden guidance (explore_hcfg.py; config chosen by 2-fold cross-fitting over base images),
R3 / R3x steering (steer_g3.py, alpha 16; ans-probe / SigLIP2-probe target), R4 thinking and R5 2x image
(explore_repair.py). Fixed = G3 error made right; broken = G3-correct item made wrong. Illusions: one view per
(image, wording), the items of the R2/R4/R5 runs. Output: results/g4_matrix.csv, results/g4_matrix_items.csv.
"""
import os, re, numpy as np, pandas as pd
from pilot import OUT

MODELS = {"qwen35_2b": "g3_heldout", "qwen35_4b": "g3_heldout", "qwen35_9b": "g3_heldout", "qwen3vl4b": "g3", "qwen3vl8b": "g3"}
SETS = ["illusions", "countbench", "mmvp"]
COLS = ["R1", "R2", "R3", "R3x", "R4", "R5", "R6x", "plain"]  # R6x (question first) and plain (its control): exploratory, not in P1
DIAG = {"BIAS": "R1", "UNSPOKEN": "R2", "INHERITED": "R4", "UNSEEN": "R5"}


def base(k):
    return re.sub(r"_px\d+", "", str(k))


def r2_crossfit(f):
    """per-item prediction of hidden guidance with (layer, gamma, scope) chosen on the other half of the base images"""
    d = np.load(f)
    names, S, y = list(d["names"]), d["S"], d["y"]
    cfg = [i for i, n in enumerate(names) if n.startswith("h_")]
    acc = (S[:, cfg].argmax(-1) == y[:, None])  # (n, n_cfg)
    u, inv = np.unique([base(k) for k in d["key"]], return_inverse=True)
    half = (np.random.default_rng(0).permutation(len(u)) % 2)[inv]
    pred, chosen = np.empty(len(y), int), {}
    for h in (0, 1):
        best = cfg[int(acc[half != h].mean(0).argmax())]
        chosen[h] = names[best]
        pred[half == h] = S[half == h, best].argmax(-1)
    return dict(zip(d["idx"].tolist(), pred.tolist())), chosen


rows, notes = [], []
for model, g3 in MODELS.items():
    for name in SETS:
        tf = f"{OUT}/g4/types_{model}_{name}.npz"
        if not os.path.exists(tf):
            notes.append(f"missing types {model} {name}"); continue
        T = np.load(tf, allow_pickle=True)
        idx, yraw, ans, typ = T["idx"], T["yraw"], T["ans"], T["type"]
        right = {"R1": T["p_cal"] == T["y"]}  # aligned with T
        st = np.load(f"{OUT}/{g3}/{model}_{name}_states.npz", allow_pickle=True)
        pos = {int(i): k for k, i in enumerate(st["idx"])}
        sel = np.array([pos[int(i)] for i in idx])
        for col, suf in (("R3", "steerA16"), ("R3x", "steerA16_ext")):
            f = f"{OUT}/{g3}/{model}_{name}_{suf}.npz"
            if os.path.exists(f):
                right[col] = np.load(f)["steer"][sel] == yraw
        maps = {}
        f = f"{OUT}/explore_hcfg/{model}_{name}.npz"
        if os.path.exists(f):
            maps["R2"], ch = r2_crossfit(f)
            notes.append(f"R2 {model} {name}: chosen {ch}")
        for col, mode in (("R4", "think"), ("R5", "res2"), ("R6x", "qfirst"), ("plain", "plain")):
            f = f"{OUT}/g4/{model}_{name}_{mode}.npz"
            if os.path.exists(f):
                d = np.load(f)
                maps[col] = dict(zip(d["idx"].tolist(), d["pred"].tolist()))
        # evaluation items: illusions = one view per (image, wording) = the items of the per-item repair runs
        keep = np.ones(len(idx), bool)
        if name == "illusions":
            one = set().union(*[set(m) for m in maps.values()]) if maps else set()
            keep = np.array([int(i) in one for i in idx])
        for col, m in maps.items():
            right[col] = np.array([m.get(int(i), -2) == yy for i, yy in zip(idx, yraw)])
            right[col + "_has"] = np.array([int(i) in m for i in idx])
        for k in np.flatnonzero(keep):
            r = dict(model=model, set=name, idx=int(idx[k]), type=typ[k], group=str(T["groups"][k]))
            for col in COLS:
                if col in right and right.get(col + "_has", np.ones(len(idx), bool))[k]:
                    r[col] = float(right[col][k])
            rows.append(r)

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/g4_matrix_items.csv", index=False)
print("\n".join(notes))
if D.empty:
    raise SystemExit("no items")
cols = [c for c in COLS if c in D]
err = D[D.type != "CORRECT"]
fix = err.groupby("type")[cols].agg(["mean", "count"])
brk = 1 - D[D.type == "CORRECT"][cols].mean()
pd.set_option("display.width", 250)
print("\nfix rate (n) per error type, pooled over models and sets")
order = [t for t in ["BIAS", "UNSPOKEN", "INHERITED", "DROPPED", "UNSEEN", "LOSS"] if t in fix.index]
tab = pd.DataFrame({c: [f"{fix.loc[t, (c, 'mean')]:.2f} ({int(fix.loc[t, (c, 'count')])})" if fix.loc[t, (c, 'count')] else "-"
                        for t in order] for c in cols}, index=order)
print(tab.to_string())
print("broken rate on correct items:", " ".join(f"{c} {brk[c]:.3f}" for c in cols))
acc = D.assign(G3=D.type == "CORRECT").groupby(["model", "set"]).agg(n=("idx", "size"), G3=("G3", "mean"), **{c: (c, "mean") for c in cols})
print("\naccuracy per cell (repair columns: on the items where the column exists)\n", acc.round(3).to_string())

# P1: diagonal dominance (R3, R3x excluded), rows with >= 15 errors pooled; permutation of row labels within cells
C = [c for c in cols if c not in ("R3", "R3x", "R6x", "plain")]
M = err[err.type.isin(DIAG)].copy()


def stat(lab):
    s, wins, rowsok = 0.0, 0, 0
    for t, dc in DIAG.items():
        g = M[lab == t]
        if len(g) < 15 or dc not in g or g[dc].notna().sum() == 0:
            continue
        rates = g[C].mean()
        rowsok += 1
        wins += int(rates[dc] >= rates.drop(dc).max())
        s += rates[dc] - rates.drop(dc).mean()
    return s, wins, rowsok


s0, wins, nrows = stat(M.type.values)
rng = np.random.default_rng(0)
cells = M.groupby(["model", "set"]).indices
null = []
for _ in range(2000):
    lab = M.type.values.copy()
    for ix in cells.values():
        lab[ix] = rng.permutation(lab[ix])
    null.append(stat(lab)[0])
p = (1 + np.sum(np.array(null) >= s0)) / (1 + len(null))
print(f"\nP1: diagonal wins {wins}/{nrows} rows (need >= 3 of 4), stat {s0:.3f}, permutation p = {p:.4f} -> "
      f"{'PASS' if wins >= 3 and p < 0.05 else 'FAIL'}")
fr = lambda t, c: err[err.type == t][c].mean() if c in err else np.nan
print(f"P2: BIAS R1 {fr('BIAS', 'R1'):.3f} >= 2 x R2 {fr('BIAS', 'R2'):.3f} -> {'PASS' if fr('BIAS', 'R1') >= 2 * fr('BIAS', 'R2') else 'FAIL'}")
print(f"P3: INHERITED R2 {fr('INHERITED', 'R2'):.3f} <= 0.15 and < UNSPOKEN R2 {fr('UNSPOKEN', 'R2'):.3f} -> "
      f"{'PASS' if fr('INHERITED', 'R2') <= 0.15 and fr('INHERITED', 'R2') < fr('UNSPOKEN', 'R2') else 'FAIL'}")
il = D[D.set == "illusions"]
if "R3x" in il:
    a = il.groupby("model")[["R3", "R3x"]].mean()
    k = int((a.R3x >= a.R3 - 0.03).sum())
    print(f"P4: illusions R3x >= R3 - 3 pts in {k}/{len(a)} models (need >= 4 of 5) -> {'PASS' if k >= 4 else 'FAIL'}\n{a.round(3).to_string()}")
print(f"P5: UNSEEN R5 {fr('UNSEEN', 'R5'):.3f} > R1 {fr('UNSEEN', 'R1'):.3f} and R2 {fr('UNSEEN', 'R2'):.3f} -> "
      f"{'PASS' if fr('UNSEEN', 'R5') > max(fr('UNSEEN', 'R1'), fr('UNSEEN', 'R2')) else 'FAIL'}")
q = err[err.model.str.startswith("qwen35") & (err.type == "INHERITED")]
if "R4" in q:
    print(f"P6: INHERITED (Qwen3.5) R4 {q.R4.mean():.3f} > R2 {q.R2.mean() if 'R2' in q else np.nan:.3f} -> "
          f"{'PASS' if 'R2' in q and q.R4.mean() > q.R2.mean() else 'FAIL/incomplete'}")
tab.to_csv(f"{OUT}/g4_matrix.csv")
