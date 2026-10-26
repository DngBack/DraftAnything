"""G7 T / Y (docs/g7_protocol.md): text-only comparison (T1-T2) and psychometric curves on illusynP (Y1-Y2; Y3 in
g7_geom.py illusynP).

Image answers: G3 decision step (results/g7_syn). Text answers: g7_text.py. Psychometric model: unpenalised logistic
regression of "unequal" (No) on |diff|, with and without regime indicators; likelihood-ratio tests (scipy chi2).
Readout curves (reported): the l0 geometric rule of g7_geom, 5-fold CV pooled over regimes, and A u B -> C.
Output: results/g7/psy.csv (per model), results/g7/psy_curves.csv (share "unequal" per model, source, regime, |diff|).
"""
import glob, os, numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from gen_calib import items
from g7_geom import rule, clean, G5SET
from pilot import OUT

D = f"{OUT}/g7"


def loglik(X, y):
    if X.shape[1] == 0:  # intercept only
        p = np.clip(y.mean(), 1e-12, 1 - 1e-12)
        return float(len(y) * (p * np.log(p) + (1 - p) * np.log(1 - p))), None
    m = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), m


def lr_test(X0, X1, y):
    l0, _ = loglik(X0, y)
    l1, m1 = loglik(X1, y)
    return chi2.sf(2 * (l1 - l0), X1.shape[1] - X0.shape[1]), m1


def curve(no, reg, diff, model, src):
    out = []
    for g in ("A", "B", "C", "all"):
        m = np.ones(len(no), bool) if g == "all" else reg == g
        for v in sorted(set(diff)):
            k = m & (diff == v)
            if k.any():
                out.append(dict(model=model, source=src, regime=g, diff=v, share_no=float(no[k].mean()), n=int(k.sum())))
    return out


def d75(no, diff):
    lv = [v for v in sorted(set(diff)) if v > 0 and no[diff == v].mean() >= 0.75]
    return lv[0] if lv else np.inf


its = items("illusynP")
diff_all = np.array([it["diff"] for it in its])
reg_all = np.array([it["regime"] for it in its])
r_all = np.array([it["logr"] for it in its])
rows, cur = [], []
for st in sorted(glob.glob(f"{OUT}/g7_syn/*_illusynP_states.npz")):
    model = os.path.basename(st)[: -len("_illusynP_states.npz")]
    if model not in G5SET:
        continue
    d = np.load(st, allow_pickle=True)
    idx, ans, y = d["idx"], d["ans"], d["y"]
    no, diff, reg = (ans == 1).astype(int), diff_all[idx], reg_all[idx]
    a = np.abs(diff)[:, None]
    dummies = np.stack([reg == "B", reg == "C"], 1).astype(float)
    p1, m1 = lr_test(np.zeros((len(no), 0)), a, no)
    p2, m2 = lr_test(a, np.hstack([a, dummies]), no)
    r = dict(model=model, n=len(idx), img_acc=np.mean(ans == y), img_fa=no[diff == 0].mean(), img_d75=d75(no, diff),
             Y1_slope=float(m1.coef_[0, 0]), Y1_p=p1, Y2_p=p2, Y2_coefB=float(m2.coef_[0, 1]), Y2_coefC=float(m2.coef_[0, 2]),
             **{f"img_acc_{g}": np.mean((ans == y)[reg == g]) for g in "ABC"}, **{f"img_fa_{g}": no[(diff == 0) & (reg == g)].mean() for g in "ABC"})
    cur += curve(no, reg, diff, model, "image")
    tf = f"{D}/text_{model}.npz"
    if os.path.exists(tf):
        T = np.load(tf)
        tno = (T["ans"] == 1).astype(int)
        tok = T["ans"] == T["y"]
        r.update(txt_acc=tok.mean(), txt_acc_eq=tok[T["diff"] == 0].mean(), txt_acc_ge10=tok[T["diff"] >= 0.10].mean(),
                 txt_acc_same=tok[idx].mean(), txt_fa=tno[T["diff"] == 0].mean(), txt_d75=d75(tno, T["diff"]))
        cur += curve(tno, T["regime"], T["diff"], model, "text")
    nf = f"{OUT}/explore_native/{model}_illusynP.npz"
    if os.path.exists(nf):
        X, rr = clean(np.load(nf)["l0"][idx]), r_all[idx]
        pred = np.zeros(len(idx), int)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            rh, tau, _ = rule(X[tr], rr[tr], X[te])
            pred[te] = np.abs(rh) > tau
        cur += curve(pred, reg, diff, model, "l0 rule (CV)")
        src, tgt = reg != "C", reg == "C"
        rh, tau, _ = rule(X[src], rr[src], X[tgt])
        cur += curve((np.abs(rh) > tau).astype(int), reg[tgt], diff[tgt], model, "l0 rule (A u B -> C)")
        r.update(l0rule_cv_acc=np.mean(pred == (rr != 0)))
    rows.append(r)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)

P = pd.DataFrame(rows).set_index("model")
os.makedirs(D, exist_ok=True)
P.to_csv(f"{D}/psy.csv")
pd.DataFrame(cur).to_csv(f"{D}/psy_curves.csv", index=False)
pd.set_option("display.width", 250)
print(P.round(4).T.to_string())


def test(name, cond, M=P):
    n, k = len(M), int(cond(M).sum())
    need = int(np.ceil(2 * n / 3))
    print(f"{name}: {k}/{n} (need {need}) -> {'PASS' if n and k >= need else 'FAIL'}")


if "txt_acc" in P:
    T = P.dropna(subset=["txt_acc"])
    test("T1 text acc >= .90 on equal and on |diff| >= .10", lambda M: (M.txt_acc_eq >= 0.90) & (M.txt_acc_ge10 >= 0.90), T)
    test("T2 text acc >= image acc + .25 (same items)", lambda M: M.txt_acc_same >= M.img_acc + 0.25, T)
test("Y1 slope on |diff| > 0, LR p < .01", lambda M: (M.Y1_slope > 0) & (M.Y1_p < 0.01))
test("Y2 regime indicators, LR p < .01", lambda M: M.Y2_p < 0.01)
