"""G5 R (docs/g5_protocol.md + Addendum 1): end-to-end routed repair on held-out base images, offline.

A router sees only label-free signals at test time: whether the cross-fitted calibrator, decision-state probe and
SigLIP2 probe agree with the model's answer. RULE is fixed; LEARNED picks per agreement pattern the repair with the
best accuracy on the training half of the base images. Output: results/g5/route.csv (per cell) and printed R1-R3.
"""
import os, numpy as np, pandas as pd
from pilot import OUT

REP = ["R2", "R4", "R5", "R1", "R3", "R3x"]  # tie-break order after "keep" (cheaper first)
D = pd.read_csv(f"{OUT}/g4_matrix_items.csv")
os.makedirs(f"{OUT}/g5", exist_ok=True)
rows = []
for (model, name), g in D.groupby(["model", "set"], sort=False):
    T = np.load(f"{OUT}/g4/types_{model}_{name}.npz", allow_pickle=True)
    pos = {int(i): k for k, i in enumerate(T["idx"])}
    s = np.array([pos[int(i)] for i in g.idx])
    u = np.unique(T["yraw"])
    ans = T["ans"][s]
    agree = lambda key: (u[T[key][s]] == ans).astype(int) if key in T.files else np.full(len(s), 2)
    pat = agree("p_cal") * 9 + agree("p_ans") * 3 + agree("p_ext")
    keep = (g.type == "CORRECT").values.astype(float)
    cols = [c for c in REP if c in g and g[c].notna().mean() >= 0.98]
    A = np.stack([keep] + [np.where(g[c].isna(), keep, g[c].values) for c in cols], 1)  # (n, 1 + n_rep): right?
    names = ["keep"] + cols
    single = A.mean(0)
    # RULE: calibrator disagrees -> R1; elif SigLIP2 disagrees -> R3x; else keep
    cal_dis, ext_dis = agree("p_cal") == 0, agree("p_ext") == 0
    rule = np.where(cal_dis, names.index("R1"), np.where(ext_dis & ("R3x" in names), names.index("R3x") if "R3x" in names else 0, 0))
    acc_rule = A[np.arange(len(A)), rule].mean()
    grp = g.group.values
    ug, inv = np.unique(grp, return_inverse=True)
    rng = np.random.default_rng(0)
    L, Rn = [], []
    for rep in range(20):
        half = (rng.permutation(len(ug)) % 2)[inv]
        accL, accR = np.zeros(len(A)), np.zeros(len(A))
        for h in (0, 1):
            tr, te = half != h, half == h
            choice = {}
            for p in np.unique(pat):
                m = tr & (pat == p)
                choice[p] = int(np.argmax(A[m].mean(0))) if m.any() else 0  # argmax keeps the first = keep / cheaper on ties
            ch = np.array([choice.get(p, 0) for p in pat[te]])
            accL[te] = A[te][np.arange(te.sum()), ch]
            freq = np.bincount(ch, minlength=len(names)) / len(ch)
            rc = rng.choice(len(names), size=te.sum(), p=freq)
            accR[te] = A[te][np.arange(te.sum()), rc]
        L.append(accL.mean()), Rn.append(accR.mean())
    r = dict(model=model, set=name, n=len(A), raw=single[0], best_single=single[1:].max(), best=names[1 + int(single[1:].argmax())],
             rule=acc_rule, learned=np.mean(L), learned_sd=np.std(L), random=np.mean(Rn), oracle=A.max(1).mean(),
             **{c: single[1 + i] for i, c in enumerate(cols)})
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/g5/route.csv", index=False)
pd.set_option("display.width", 250)
print((R.set_index(["model", "set"]).drop(columns=["n"]).select_dtypes("number") * 100).round(1).assign(best=R.best.values).to_string())
d1 = (R.learned - R.best_single).mean() * 100
print(f"\nR1: LEARNED - best single, pooled = {d1:+.1f} pts (need >= +1) -> {'PASS' if d1 >= 1 else 'FAIL'}")
k = int((R.learned > R.raw).sum())
print(f"R2: LEARNED > raw in {k}/{len(R)} cells (need >= {int(np.ceil(2 * len(R) / 3))}) -> {'PASS' if k >= np.ceil(2 * len(R) / 3) else 'FAIL'}")
q = R[R.set != "mmvp"]
k3 = int((q.rule >= q.best_single).sum())
print(f"R3: RULE >= best single in {k3}/{len(q)} illusion+CountBench cells (need >= {int(np.ceil(len(q) / 2))}) -> {'PASS' if k3 >= np.ceil(len(q) / 2) else 'FAIL'}")
print("pooled means:", (R[["raw", "best_single", "rule", "learned", "random", "oracle"]].mean() * 100).round(1).to_dict())
print("by set:\n", (R.groupby("set")[["raw", "best_single", "rule", "learned", "random", "oracle"]].mean() * 100).round(1).to_string())
