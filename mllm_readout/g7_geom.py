"""G7 G / Y3 (docs/g7_protocol.md): geometric readout of r = log(L_ref / L_other) with a source-fixed decision rule.

Ridge on standardised features (alpha by shuffled 5-fold CV in the source), predict "equal" iff |r_hat| <= tau, tau
maximising equal/unequal accuracy of the 5-fold CV predictions in the source; applied unchanged to the target.
Comparator: the direct equal/unequal linear classifier (steer_g3.fit_probe), same source and target.
Sets: illusynC (primary, tests G1-G4), illusynL / illusyn (reported), illusynP (Y3: source A u B, target C).
Readouts: own LLM input l0 and encoder output pre (explore_native), decision state at 75% (G3 states), SigLIP2 /
DINOv2 grid (explore_input), pixel statistics; Addendum A: own LLM input as a 4x4 grid (explore_input --src <model>).
Output: results/g7/geom_<set>.csv.   usage: python g7_geom.py [illusynC | illusynL | illusyn | illusynP]
"""
import glob, os, sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from steer_g3 import fit_probe
from gen_calib import items
from illusyn import pixel_stats
from pilot import OUT

G5SET = ["qwen3vl4b", "qwen3vl8b", "qwen35_4b", "qwen35_9b", "gemma3_4b", "internvl35_8b", "llava15_7b", "qwen2vl_7b"]
ALPHAS = (1, 10, 100, 1000, 10000)
SYN = {"illusynC": "g6_syn", "illusynL": "g5_syn", "illusyn": "g5_syn", "illusynP": "g7_syn"}


def clean(X):
    return np.clip(np.nan_to_num(np.asarray(X, np.float32), posinf=0, neginf=0), -1e4, 1e4)


def best_tau(rh, eq):
    """threshold on |r_hat| maximising accuracy of 'equal iff |r_hat| <= tau' (midpoints between sorted values)"""
    a = np.sort(np.abs(rh))
    cand = np.concatenate([[0.0], (a[1:] + a[:-1]) / 2, [a[-1] + 1e-6]])
    acc = [np.mean((np.abs(rh) <= t) == eq) for t in cand]
    return float(cand[int(np.argmax(acc))])


def rule(Xs, rs, Xt):
    """fit on the source; returns r_hat on the target, tau (source CV), alpha"""
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-4
    Zs, Zt = (Xs - mu) / sd, (Xt - mu) / sd
    kf = KFold(5, shuffle=True, random_state=0)
    err = {a: np.mean((cross_val_predict(Ridge(alpha=a), Zs, rs, cv=kf) - rs) ** 2) for a in ALPHAS}
    al = min(err, key=err.get)
    rcv = cross_val_predict(Ridge(alpha=al), Zs, rs, cv=kf)
    tau = best_tau(rcv, rs == 0)
    return Ridge(alpha=al).fit(Zs, rs).predict(Zt), tau, al


def evaluate(X, r, src, tgt):
    """rule vs binary classifier from src mask to tgt mask; plus reported quantities on the target"""
    rh, tau, al = rule(X[src], r[src], X[tgt])
    rt, eq = r[tgt], r[tgt] == 0
    y = (r != 0).astype(int)
    u = ~eq
    return dict(rule=np.mean((np.abs(rh) <= tau) == eq), binary=np.mean(fit_probe(X[src], y[src])(X[tgt]) == y[tgt]),
                corr=np.corrcoef(rh, rt)[0, 1], signed=np.mean(np.sign(rh[u]) == np.sign(rt[u])),
                oracle=np.mean((np.abs(rh) <= best_tau(rh, eq)) == eq), tau=tau, alpha=al, say_equal=np.mean(np.abs(rh) <= tau))


def across(X, r, reg, pairs):
    res = [evaluate(X, r, np.isin(reg, list(s)), np.isin(reg, list(t))) for s, t in pairs]
    return {k: float(np.mean([z[k] for z in res])) for k in res[0]}


if __name__ == "__main__":
    SET = sys.argv[1] if len(sys.argv) > 1 else "illusynC"
    PAIRS = [("AB", "C")] if SET == "illusynP" else [("A", "B"), ("B", "A")]
    its = items(SET)
    r_all = np.array([it["logr"] for it in its])
    reg_all = np.array([it["regime"] for it in its])
    rows = []
    feats = {"pixel": np.stack([pixel_stats(it["image"]) for it in its])}
    for k in ("siglip2", "dinov2"):
        f = f"{OUT}/explore_input/{k}_{SET}.npz"
        if os.path.exists(f):
            feats[k] = clean(np.load(f)["feats"][:, 0].reshape(len(its), -1))
    for k, X in feats.items():
        rows.append(dict(model="(model-free)", readout=k, n=len(its), **across(X, r_all, reg_all, PAIRS)))
        print(rows[-1], flush=True)
    for st in sorted(glob.glob(f"{OUT}/{SYN[SET]}/*_{SET}_states.npz")):
        model = os.path.basename(st)[: -len(f"_{SET}_states.npz")]
        if model not in G5SET:
            continue
        d = np.load(st, allow_pickle=True)
        idx = d["idx"]
        r, reg = r_all[idx], reg_all[idx]
        F = {"dec": clean(d["last"][:, round(0.75 * (d["last"].shape[1] - 1)) + 1])}
        nf = f"{OUT}/explore_native/{model}_{SET}.npz"
        if os.path.exists(nf):
            N = np.load(nf)
            F["l0"], F["pre"] = clean(N["l0"][idx]), clean(N["pre"][idx])
        gf = f"{OUT}/explore_input/{model}_{SET}.npz"  # Addendum A: own LLM input as a 4x4 grid + mean (layer 0)
        if os.path.exists(gf):
            F["l0grid"] = clean(np.load(gf)["feats"][idx][:, 0].reshape(len(idx), -1))
        raw = np.mean(d["ans"] == d["y"])
        for k, X in F.items():
            rows.append(dict(model=model, readout=k, n=len(idx), model_acc=raw, **across(X, r, reg, PAIRS)))
            print({a: round(b, 3) if isinstance(b, float) else b for a, b in rows[-1].items()}, flush=True)
    G = pd.DataFrame(rows)
    os.makedirs(f"{OUT}/g7", exist_ok=True)
    G.to_csv(f"{OUT}/g7/geom_{SET}.csv", index=False)
    pd.set_option("display.width", 250)
    print(G.round(3).to_string(index=False))

    def test(name, rd, cond):
        M = G[(G.readout == rd) & G.model.isin(G5SET)]
        k, n = int(cond(M).sum()), len(M)
        need = int(np.ceil(2 * n / 3))
        print(f"{name}: {k}/{n} (need {need}) -> {'PASS' if n and k >= need else 'FAIL'}")

    if SET == "illusynC":
        test("G1 l0 rule across >= binary across + .05", "l0", lambda M: M.rule >= M.binary + 0.05)
        test("G2 decision rule across >= binary across + .05", "dec", lambda M: M.rule >= M.binary + 0.05)
        test("G3 l0 rule across >= .60", "l0", lambda M: M.rule >= 0.60)
        s = G[(G.model == "(model-free)") & (G.readout == "siglip2")].iloc[0]
        print(f"G4 SigLIP2 rule {s.rule:.3f} >= binary {s.binary:.3f} + .05 -> {'PASS' if s.rule >= s.binary + 0.05 else 'FAIL'}")
    if "l0grid" in set(G.readout):
        test("Addendum A (exploratory): grid l0 rule >= binary + .05 and >= .60", "l0grid", lambda M: (M.rule >= M.binary + 0.05) & (M.rule >= 0.60))
    if SET == "illusynP":
        test("Y3 l0 rule (A u B -> C) >= binary + .05", "l0", lambda M: M.rule >= M.binary + 0.05)
