"""G6 P (docs/g6_protocol.md): the price of a repair, predicted by the bracketed diagnostic. Tests P7-P10.

Per model x set cell (>= 30 errors), base images are split in half (seed 0). Half A: error types with probes
cross-fitted within A (calibrator on centred candidate scores, decision state at 75% depth, E = own LLM input l0).
Half B: price of each readout = smallest k in KS labelled items per class (per wording on illusions) at which the
ridge readout's mean accuracy over 30 draws is >= raw + 5 points (inf if none); draws train on a random half of B's
images (min(k, available) items per class, every class kept) and test on the other half.
Readouts: cal (candidate scores), inp (own LLM input, explore_native l0), dec (decision state), enc (SigLIP2 mean).
Output: results/g6/price.csv (one row per cell) and results/g6/price_curves.csv (accuracy per k).
"""
import glob, os, re, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifier
from pilot import OUT, probe_cv

KS = [1, 2, 4, 8, 16, 32]
SETS = ["illusions", "countbench", "dice", "tally", "carlogos", "illusynC"]
DIRS = {s: ("g6_syn",) if s == "illusynC" else ("g3", "g3_heldout") for s in SETS}


def clean(X):
    return np.clip(np.nan_to_num(np.asarray(X, np.float32), posinf=0, neginf=0), -1e4, 1e4)


def ridge(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    return RidgeClassifier(alpha=10.0).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)


def load(p):
    model, name = re.match(r"(.+)_([A-Za-z]+)_states\.npz", os.path.basename(p)).groups()
    d = np.load(p, allow_pickle=True)
    idx, y = d["idx"], np.asarray(d["y"])
    L = round(0.75 * (d["last"].shape[1] - 1))
    sc = clean(d["scores"])
    F = {"cal": sc - sc.mean(1, keepdims=True), "dec": clean(d["last"][:, L + 1])}
    nf, ef = f"{OUT}/explore_native/{model}_{name}.npz", f"{OUT}/explore_input/siglip2_{name}.npz"
    if os.path.exists(nf):
        F["inp"] = clean(np.load(nf)["l0"][idx])
    if os.path.exists(ef):
        F["enc"] = clean(np.load(ef)["feats"][idx][:, 0, 16])
    g = np.array([re.sub(r"_px\d+", "", str(k)) for k in d["key"]])
    q = np.asarray(d["q"]) if name == "illusions" else np.array([""] * len(y))
    return model, name, F, y, np.asarray(d["ans"]), g, q


def curves(F, y, ans, g, q, B):
    out = []
    for k in KS:
        acc = {f: [] for f in F}
        raw = []
        for draw in range(30):
            rng = np.random.default_rng(1000 * k + draw)
            hit = {f: [0, 0] for f in F}
            rr = [0, 0]
            for qq in np.unique(q[B]):
                m = B & (q == qq)
                u = np.unique(g[m])
                trg = set(rng.permutation(u)[: len(u) // 2])
                trm, tem = m & np.isin(g, list(trg)), m & ~np.isin(g, list(trg))
                cls = [c for c in np.unique(y[m]) if (trm & (y == c)).any()]
                if len(cls) < 2 or not tem.any():
                    continue
                tr = np.concatenate([rng.choice(np.flatnonzero(trm & (y == c)), min(k, int((trm & (y == c)).sum())), replace=False) for c in cls])
                te = np.flatnonzero(tem)
                rr[0] += int((ans[te] == y[te]).sum()); rr[1] += len(te)
                for f, X in F.items():
                    hit[f][0] += int((ridge(X[tr], y[tr], X[te]) == y[te]).sum()); hit[f][1] += len(te)
            if rr[1]:
                raw.append(rr[0] / rr[1])
                for f in F:
                    acc[f].append(hit[f][0] / hit[f][1])
        out.append(dict(k=k, raw=np.mean(raw), **{f: np.mean(v) for f, v in acc.items()}))
    return out


cells = []
for s in SETS:
    for dd in DIRS[s]:
        cells += sorted(glob.glob(f"{OUT}/{dd}/*_{s}_states.npz"))
data = [load(p) for p in cells]
data = [x for x in data if (x[4] != x[3]).sum() >= 30 and "inp" in x[2]]
SMOKE = os.environ.get("SMOKE") == "1"  # code check only: two cells, k <= 2, no verdicts
if SMOKE:
    data, KS = data[:2], KS[:2]

rows, todo = [], []
for model, name, F, y, ans, g, q in data:
    u = np.unique(g)
    A = np.isin(g, np.random.default_rng(0).permutation(u)[: len(u) // 2])
    ok = {f: probe_cv(F[f][A], y[A], groups=g[A])[0] == y[A] for f in ("cal", "dec", "inp")}
    err = ans[A] != y[A]
    C, Dd, E = ok["cal"], ok["dec"], ok["inp"]
    ty = dict(BIAS=err & C, UNSPOKEN=err & ~C & Dd & ~E, INHERITED=err & ~C & Dd & E, DROPPED=err & ~C & ~Dd & E, UNSEEN=err & ~C & ~Dd & ~E)
    r = dict(model=model, set=name, n=len(y), errors=int((ans != y).sum()), errors_A=int(err.sum()),
             **{f"share_{t}": v.sum() / max(err.sum(), 1) for t, v in ty.items()})
    rows.append(r)
    todo.append((F, y, ans, g, q, ~A))

res = Parallel(n_jobs=16)(delayed(curves)(*t) for t in todo)
cur = []
for r, cv in zip(rows, res):
    for c in cv:
        cur.append(dict(model=r["model"], set=r["set"], **c))
    for f in ("cal", "inp", "dec", "enc"):
        ks = [c["k"] for c in cv if f in c and c[f] >= c["raw"] + 0.05]
        r[f"price_{f}"] = min(ks) if ks else np.inf
        r[f"acc8_{f}"] = next((c[f] for c in cv if c["k"] == 8 and f in c), np.nan)
    r["raw_B"] = np.mean([c["raw"] for c in cv])

P = pd.DataFrame(rows)
if SMOKE:
    print(P.columns.tolist(), len(P), len(cur))
    raise SystemExit
os.makedirs(f"{OUT}/g6", exist_ok=True)
P.to_csv(f"{OUT}/g6/price.csv", index=False)
pd.DataFrame(cur).to_csv(f"{OUT}/g6/price_curves.csv", index=False)
pd.set_option("display.width", 250)
print(P.round(3).to_string())


def test(name, m, cond):
    n, k = int(m.sum()), int((m & cond).sum())
    need = int(np.ceil(2 * n / 3))
    print(f"{name}: {k}/{n} (need {need}) -> {'PASS' if n and k >= need else ('VACUOUS' if not n else 'FAIL')}")


sep = P.share_INHERITED + P.share_DROPPED
test("P7 BIAS >= 1/3 -> price(cal) <= 8", P.share_BIAS >= 1 / 3, P.price_cal <= 8)
test("P8 BIAS < .10 -> price(cal) = inf", P.share_BIAS < 0.10, np.isinf(P.price_cal))
test("P9 INH+DROP >= 1/2 -> price(inp) finite and <= price(dec)", sep >= 0.5, np.isfinite(P.price_inp) & (P.price_inp <= P.price_dec))
test("P10 UNSEEN >= 1/2 -> cal, inp, dec all inf", P.share_UNSEEN >= 0.5,
     np.isinf(P.price_cal) & np.isinf(P.price_inp) & np.isinf(P.price_dec))
print(P.replace(np.inf, 99).groupby("set")[[c for c in P.columns if c.startswith(("price_", "acc8_"))] + ["raw_B"]].median().round(3).to_string())
