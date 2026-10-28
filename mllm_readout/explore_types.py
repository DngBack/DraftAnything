"""G4 diagnostic (docs/g4_protocol.md): per-error types with an input-side bracket, and checks I1-I3.

For every G3 cell (results/g3, results/g3_heldout) on a question-fixed set, all predictors are cross-fitted (5 folds
grouped by base image): cal (affine calibrator on candidate logits), ans (probe on the decision-token state at 75% depth;
also the best layer, reported only), ext (SigLIP2 4x4+mean grid, primary), dino (DINOv2 grid), mllm0/mllm4 (projector
output / layer-4 grid of the same MLLM), and ext_shuf (ext probe trained on permuted labels, 5 permutations: the chance
rate at which an input probe "recovers" an answer). MMVP gets cal/ans only (the answer depends on the question).
Output: results/g4_types.csv (per cell) and results/g4/types_<model>_<set>.npz (per item, aligned by idx).
"""
import glob, os, re, numpy as np, pandas as pd
from pilot import probe_cv, OUT

QFIXED = ["illusions", "countbench", "carlogos", "dice", "tally"]
INP = f"{OUT}/explore_input"
os.makedirs(f"{OUT}/g4", exist_ok=True)


def clean(X):
    return np.clip(np.nan_to_num(X.astype(np.float32), posinf=0, neginf=0), -1e4, 1e4)


rows = []
paths = sorted(glob.glob(f"{OUT}/g3/*_states.npz")) + sorted(glob.glob(f"{OUT}/g3_heldout/*_states.npz"))
for p in paths:
    model, name = re.match(r"(.+)_([a-z]+)_states\.npz", os.path.basename(p)).groups()
    if name not in QFIXED + ["mmvp"]:
        continue
    d = np.load(p, allow_pickle=True)
    y, ans, sc, idx = d["y"], d["ans"], np.asarray(d["scores"], np.float32), d["idx"]
    ok = np.isin(y, np.flatnonzero(np.bincount(y) >= 5))  # classes with < 5 items dropped (as the paper's cells)
    y, ans, sc, idx = y[ok], ans[ok], sc[ok], idx[ok]
    groups = np.array([re.sub(r"_px\d+", "", str(k)) for k in d["key"][ok]])
    yy = np.unique(y, return_inverse=True)[1]
    nL = d["last"].shape[1] - 1
    L = round(0.75 * nL)
    P = {"cal": probe_cv(sc - sc.mean(1, keepdims=True), yy, groups=groups)[0]}
    allL = probe_cv(clean(d["last"][ok]), yy, groups=groups)[0]
    P["ans"], P["ans_best"] = allL[L + 1], allL[int((allL == yy).mean(1).argmax())]
    if name in QFIXED:
        for src, key in (("siglip2", "ext"), ("dinov2", "dino"), (model, "mllm")):
            f = f"{INP}/{src}_{name}.npz"
            if not os.path.exists(f):
                continue
            F = np.load(f)["feats"][idx].astype(np.float32)  # (n, n_layers, 17, d)
            if key == "mllm":
                P["mllm0"] = probe_cv(clean(F[:, 0].reshape(len(yy), -1)), yy, groups=groups)[0]
                P["mllm4"] = probe_cv(clean(F[:, 1].reshape(len(yy), -1)), yy, groups=groups)[0]
            else:
                E = F[:, 0].reshape(len(yy), -1)
                P[key] = probe_cv(E, yy, groups=groups)[0]
                P[key + "_mean"] = probe_cv(F[:, 0, 16], yy, groups=groups)[0]
                if key == "ext":
                    rng = np.random.default_rng(0)
                    P["ext_shuf"] = np.stack([probe_cv(E, rng.permutation(yy), groups=groups)[0] for _ in range(5)])
        f = f"{INP}_deep/{model}_{name}.npz"  # matched control: the same model's image tokens at the probe's depth (question-blind)
        if os.path.exists(f):
            P["mllmL"] = probe_cv(clean(np.load(f)["feats"][idx][:, 0].reshape(len(yy), -1)), yy, groups=groups)[0]
    if name == "illusions":  # leave-one-illusion-out: does a probe trained on 5 illusion types transfer to the 6th?
        sub = d["group"][ok]
        X = {"ans": clean(d["last"][ok][:, L + 1])}
        for k, f in (("ext", f"{INP}/siglip2_{name}.npz"), ("mllm0", f"{INP}/{model}_{name}.npz"), ("mllmL", f"{INP}_deep/{model}_{name}.npz")):
            if os.path.exists(f):
                X[k] = clean(np.load(f)["feats"][idx][:, 0].reshape(len(yy), -1))
        for k, v in X.items():
            P[f"loso_{k}"] = probe_cv(v, yy, k=len(np.unique(sub)), groups=sub)[0]
    R = {k: v == yy for k, v in P.items()}
    err = ans != y
    t = np.full(len(y), "", dtype=object)
    t[~err] = "CORRECT"
    t[err & R["cal"]] = "BIAS"
    loss = err & ~R["cal"] & R["ans"]
    if "ext" in R:
        t[loss & ~R["ext"]], t[loss & R["ext"]] = "UNSPOKEN", "INHERITED"
        t[err & ~R["cal"] & ~R["ans"] & R["ext"]] = "DROPPED"
        t[err & ~R["cal"] & ~R["ans"] & ~R["ext"]] = "UNSEEN"
    else:
        t[loss], t[err & ~R["cal"] & ~R["ans"]] = "LOSS", "UNSEEN"
    np.savez(f"{OUT}/g4/types_{model}_{name}.npz", idx=idx, y=yy, yraw=y, ans=ans, type=t.astype(str), groups=groups, layer=L,
             **{f"p_{k}": v for k, v in P.items()})
    r = dict(model=model, set=name, held="heldout" in p, n=len(y), errors=int(err.sum()), raw=np.mean(~err),
             **{f"acc_{k}": float(np.mean(v)) for k, v in R.items()},
             **{k: int((t == k).sum()) for k in ("BIAS", "UNSPOKEN", "INHERITED", "DROPPED", "UNSEEN", "LOSS")})
    if "ext" in R:
        r["loss_n"] = int(loss.sum())
        r["loss_ext"] = float(R["ext"][loss].mean()) if loss.any() else np.nan
        r["loss_ext_shuf"] = float(R["ext_shuf"][:, loss].mean()) if loss.any() else np.nan
        r["loss_ext_n"] = int(R["ext"][loss].sum())
        r["loss_shuf_n"] = float(R["ext_shuf"][:, loss].sum(1).mean())
    rows.append(r)
    print(f"{model:14s} {name:10s} n={len(y)} raw {r['raw']:.2f} cal {r['acc_cal']:.2f} ans {r['acc_ans']:.2f} "
          f"ext {r.get('acc_ext', np.nan):.2f} dino {r.get('acc_dino', np.nan):.2f} mllm0 {r.get('acc_mllm0', np.nan):.2f} "
          f"| B/U/I/D/S {r['BIAS']}/{r['UNSPOKEN']}/{r['INHERITED']}/{r['DROPPED']}/{r['UNSEEN']}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/g4_types.csv", index=False)

G3M = ["gemma3_27b", "gemma3_4b", "internvl35_8b", "llava15_7b", "qwen2vl_7b", "qwen3vl2b", "qwen3vl32b", "qwen3vl4b", "qwen3vl8b"]
m = df[~df.held & df.model.isin(G3M)].set_index(["set", "model"])
for s, rule, thr in (("illusions", lambda r: r.acc_ext >= r.acc_ans - 0.05, 7), ("countbench", lambda r: r.acc_ans >= r.acc_ext + 0.20, 7)):
    if s in m.index.get_level_values(0) and "acc_ext" in m:
        c = m.loc[s].dropna(subset=["acc_ext"])
        k = int(sum(rule(r) for _, r in c.iterrows()))
        print(f"{'I1' if s == 'illusions' else 'I2'} {s}: {k}/{len(c)} models (need >= {thr} of 9) -> {'PASS' if k >= thr and len(c) == 9 else 'FAIL/incomplete'}")
if "loss_n" in df:
    il = df[(df.set == "illusions") & ~df.held & df.model.isin(G3M)].dropna(subset=["loss_ext"])
    a, b = il.loss_ext_n.sum() / il.loss_n.sum(), il.loss_shuf_n.sum() / il.loss_n.sum()
    print(f"I3 illusions pooled LOSS n={il.loss_n.sum()}: ext right {a:.3f} vs shuffled {b:.3f} -> diff {100 * (a - b):.1f} pts "
          f"({'PASS' if a - b >= 0.25 else 'FAIL'}; models {len(il)})")
