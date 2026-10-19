"""G5 N (docs/g5_protocol.md): input control along each MLLM's own visual pathway, tests N1-N5.

Features (question-blind, explore_native.py): pre = input of the model's projector (vision-encoder output), mean over
patches; l0 = projector output entering LLM layer 0, mean over image tokens. Decision state = G3 state at 75% depth.
All probes cross-fitted with the folds of explore_types.py (5, grouped by base image). Types are recomputed with
E_native = l0 probe, with the G4 calibrator and decision-state probe. N5 repeats N1/N2 with 256 PCA dims fitted per fold.
Output: results/g5/native.csv
"""
import glob, os, numpy as np, pandas as pd, torch
from pilot import probe_cv, OUT
from steer_g3 import fit_probe

G5SET = ["qwen3vl4b", "qwen3vl8b", "qwen35_4b", "qwen35_9b", "gemma3_4b", "internvl35_8b", "llava15_7b", "qwen2vl_7b"]
NAT = f"{OUT}/explore_native"


def clean(X):
    return np.clip(np.nan_to_num(X.astype(np.float32), posinf=0, neginf=0), -1e4, 1e4)


def folds(groups, k=5, seed=0):  # as pilot.probe_cv
    u, inv = np.unique(groups, return_inverse=True)
    return (torch.randperm(len(u), generator=torch.Generator().manual_seed(seed)) % k).numpy()[inv]


def probe_pca(X, y, groups, dims=256):
    fold, pred = folds(groups), np.empty(len(y), int)
    for f in range(5):
        tr, te = fold != f, fold == f
        A = torch.as_tensor(X[tr], device="cuda")
        mu, sd = A.mean(0), A.std(0) + 1e-4
        V = torch.linalg.svd((A - mu) / sd, full_matrices=False)[2][:dims].T
        proj = lambda Z: (((torch.as_tensor(Z, device="cuda") - mu) / sd) @ V).cpu().numpy()
        pred[te] = fit_probe(proj(X[tr]), y[tr])(proj(X[te]))
    return pred


rows = []
for tf in sorted(glob.glob(f"{OUT}/g4/types_*_illusions.npz") + glob.glob(f"{OUT}/g4/types_*_countbench.npz")):
    model, name = os.path.basename(tf)[6:-4].rsplit("_", 1)
    nf = f"{NAT}/{model}_{name}.npz"
    st = (glob.glob(f"{OUT}/g3/{model}_{name}_states.npz") + glob.glob(f"{OUT}/g3_heldout/{model}_{name}_states.npz"))
    if not os.path.exists(nf) or not st:
        continue
    T, N = np.load(tf, allow_pickle=True), np.load(nf)
    d = np.load(st[0], allow_pickle=True)
    pos = {int(i): k for k, i in enumerate(d["idx"])}
    s = np.array([pos[int(i)] for i in T["idx"]])
    y, g, L = T["y"], T["groups"], int(T["layer"])
    Hd = clean(d["last"][s, L + 1])
    pre, l0 = clean(N["pre"][T["idx"]]), clean(N["l0"][T["idx"]])
    P = {"pre": probe_cv(pre, y, groups=g)[0], "l0": probe_cv(l0, y, groups=g)[0],
         "l0+dec": probe_cv(np.concatenate([l0, Hd], 1), y, groups=g)[0]}
    for k, X in (("dec", Hd), ("l0", l0), ("pre", pre)):
        P[k + "_pca"] = probe_pca(X, y, g)
    acc = {k: np.mean(v == y) for k, v in P.items()}
    for k in ("cal", "ans", "ext", "mllm0", "mllmL"):
        if f"p_{k}" in T.files:
            acc[k] = np.mean(T[f"p_{k}"] == y)
    err = T["ans"] != T["yraw"]
    C, D, E = T["p_cal"] == y, T["p_ans"] == y, P["l0"] == y
    r = dict(model=model, set=name, n=len(y), errors=int(err.sum()), raw=1 - err.mean(), **acc,
             va_l0=acc["l0+dec"] - acc["l0"],
             BIAS=int((err & C).sum()), UNSPOKEN=int((err & ~C & D & ~E).sum()), INHERITED=int((err & ~C & D & E).sum()),
             DROPPED=int((err & ~C & ~D & E).sum()), UNSEEN=int((err & ~C & ~D & ~E).sum()))
    rows.append(r)
    print(model, name, {k: round(float(v), 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

R = pd.DataFrame(rows)
os.makedirs(f"{OUT}/g5", exist_ok=True)
R.to_csv(f"{OUT}/g5/native.csv", index=False)
pd.set_option("display.width", 250)
cols = ["raw", "cal", "ans", "ext", "pre", "l0", "mllm0", "mllmL", "l0+dec", "va_l0", "dec_pca", "l0_pca", "pre_pca"]
for name in ("illusions", "countbench"):
    print(f"\n{name}\n", (R[R.set == name].set_index("model")[[c for c in cols if c in R]] * 100).round(1).to_string())
    print(R[R.set == name].set_index("model")[["errors", "BIAS", "UNSPOKEN", "INHERITED", "DROPPED", "UNSEEN"]].to_string())
G = R[R.model.isin(G5SET)]
il, cb = G[G.set == "illusions"].set_index("model"), G[G.set == "countbench"].set_index("model")
need = lambda df: int(np.ceil(2 * len(df) / 3))
verdict = lambda k, df: f"{k}/{len(df)} (need {need(df)}) -> {'PASS' if k >= need(df) else 'FAIL'}"
print("\nG5 tests on", len(il), "illusion and", len(cb), "CountBench models")
print("N1 illusions l0 >= dec - 5:", verdict(int((il.l0 >= il.ans - 0.05).sum()), il))
print("N2 CountBench dec >= l0 + 20:", verdict(int((cb.ans >= cb.l0 + 0.20).sum()), cb))
print("N3 illusions INHERITED > UNSPOKEN (E_native):", verdict(int((il.INHERITED > il.UNSPOKEN).sum()), il))
k4 = int(((il.va_l0 <= 0.02) & True).sum()), int((cb.va_l0 >= 0.15).sum())
print("N4 value added over l0 <= +2 illusions:", verdict(k4[0], il), "| >= +15 CountBench:", verdict(k4[1], cb))
print("N5 (PCA 256) illusions l0 >= dec - 5:", verdict(int((il.l0_pca >= il.dec_pca - 0.05).sum()), il),
      "| CountBench dec >= l0 + 20:", verdict(int((cb.dec_pca >= cb.l0_pca + 0.20).sum()), cb))
