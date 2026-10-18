"""Exploratory (2026-10-18): the label cost of a repair. How many labelled items per class does a readout need?

Per cell and question (illusions: each of the 12 wordings; other sets: one question), k labelled items per class are
drawn from a random half of the base images and the readout is scored on the other half (30 draws). Readouts:
prototype classifier (nearest class mean after centring on the training mean; the classifier implied by class-mean
steering) and ridge on standardised features. Features: cal (candidate scores), ans (decision state at 75% depth),
ext (SigLIP2 mean), mllm0 (projector-output mean). Output: results/explore_labelcost.csv
"""
import glob, os, re, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifier
from pilot import OUT

KS = [1, 2, 4, 8, 16, 32]
SETS = ["illusions", "countbench", "dice", "tally"]


def proto(Xtr, ytr, Xte):
    mu = Xtr.mean(0)
    C = np.stack([Xtr[ytr == c].mean(0) - mu for c in np.unique(ytr)])
    return np.unique(ytr)[((Xte - mu) @ C.T - 0.5 * (C ** 2).sum(1)).argmax(1)]


def ridge(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    return RidgeClassifier(alpha=10.0).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)


def cell(p):
    model, name = re.match(r"(.+)_([a-z]+)_states\.npz", os.path.basename(p)).groups()
    d = np.load(p, allow_pickle=True)
    y, idx, q = d["y"], d["idx"], d["q"]
    g = np.array([re.sub(r"_px\d+", "", str(k)) for k in d["key"]])
    L = round(0.75 * (d["last"].shape[1] - 1))
    F = {"cal": np.asarray(d["scores"], np.float32),
         "ans": np.clip(np.nan_to_num(d["last"][:, L + 1].astype(np.float32)), -1e4, 1e4)}
    for k, f in (("ext", f"{OUT}/explore_input/siglip2_{name}.npz"), ("mllm0", f"{OUT}/explore_input/{model}_{name}.npz")):
        if os.path.exists(f):
            F[k] = np.clip(np.nan_to_num(np.load(f)["feats"][idx][:, 0, 16].astype(np.float32)), -1e4, 1e4)
    out = []
    qs = np.unique(q) if name == "illusions" else [None]
    for k in KS:
        for draw in range(30):
            rng = np.random.default_rng(1000 * k + draw)
            res = {f"{f}_{r}": [0, 0] for f in F for r in ("proto", "ridge")}
            raw = [0, 0]
            for qq in qs:
                m = np.ones(len(y), bool) if qq is None else q == qq
                u = np.unique(g[m])
                tr_g = set(rng.permutation(u)[: len(u) // 2])
                trm = m & np.isin(g, list(tr_g))
                tem = m & ~np.isin(g, list(tr_g))
                cls = [c for c in np.unique(y[m]) if (trm & (y == c)).sum() >= k and (tem & (y == c)).any()]
                if len(cls) < 2:
                    continue
                tr = np.concatenate([rng.choice(np.flatnonzero(trm & (y == c)), k, replace=False) for c in cls])
                te = np.flatnonzero(tem & np.isin(y, cls))
                raw[0] += int((d["ans"][te] == y[te]).sum()); raw[1] += len(te)
                for f, X in F.items():
                    for r, fn in (("proto", proto), ("ridge", ridge)):
                        pr = fn(X[tr], y[tr], X[te])
                        res[f"{f}_{r}"][0] += int((pr == y[te]).sum()); res[f"{f}_{r}"][1] += len(te)
            if raw[1]:
                out.append(dict(model=model, set=name, k=k, draw=draw, raw=raw[0] / raw[1],
                                **{key: a / b for key, (a, b) in res.items() if b}))
    return out


paths = [p for dd in ("g3", "g3_heldout") for p in sorted(glob.glob(f"{OUT}/{dd}/*_states.npz"))
         if re.match(r".+_([a-z]+)_states", p).group(1) in SETS]
rows = [r for rr in Parallel(n_jobs=16)(delayed(cell)(p) for p in paths) for r in rr]
D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/explore_labelcost.csv", index=False)
cols = [c for c in D.columns if c.endswith(("_proto", "_ridge"))]
pd.set_option("display.width", 250)
print(D.groupby(["set", "k"])[["raw"] + sorted(cols)].mean().round(3).to_string())
