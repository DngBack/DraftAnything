"""G6 M' (docs/g6_protocol.md): label-specific interchange effect, tests M4-M6.

Delta(l) = P(read-out = partner's label | other-label swap at l) - P(read-out = the other label | same-label swap at l),
for the decision-state probe (cross-fitted as in g5_swap.analyze) and for the model's answer. Depth 0 from stored G3
states (the swap there equals running the partner); 12.5 / 25 / 50 / 75% from results/g5/swap_<model>.npz, with
same-label swaps at 25 / 75% there and at 12.5 / 50% in results/g5/swapctrl_<model>.npz.
Output: results/g6/swap_delta.csv
"""
import glob, os, numpy as np, pandas as pd
from pilot import OUT
from steer_g3 import fit_probe
from g5_swap import meta, FR

G5SET = ["qwen3vl4b", "qwen3vl8b", "qwen35_4b", "qwen35_9b", "gemma3_4b", "internvl35_8b", "llava15_7b", "qwen2vl_7b"]
UNSEEN = ["qwen35_4b", "internvl35_8b", "qwen2vl_7b"]
D = f"{OUT}/g5"


def depth(name, nL):
    return min(FR, key=lambda f: abs(f - int(name[4:]) / nL))


rows = []
for f in sorted(glob.glob(f"{D}/swap_*.npz")):
    model = os.path.basename(f)[5:-4]
    if model not in G5SET:
        continue
    S = np.load(f)
    fc = f"{D}/swapctrl_{model}.npz"
    Sc = np.load(fc) if os.path.exists(fc) else None
    st = (glob.glob(f"{OUT}/g3/{model}_illusions_states.npz") + glob.glob(f"{OUT}/g3_heldout/{model}_illusions_states.npz"))[0]
    d = np.load(st, allow_pickle=True)
    L, y, ans, nL = int(S["L"]), d["y"], d["ans"], int(S["nL"])
    X = d["last"][:, L + 1].astype(np.float32)
    g = np.array(["_".join(meta(k, "")[:2]) for k in d["key"]])
    ug, inv = np.unique(g, return_inverse=True)
    fold = (np.random.default_rng(0).permutation(len(ug)) % 5)[inv]
    srcs = [(S, c) for c in range(len(S["conds"]))] + ([(Sc, c) for c in range(len(Sc["conds"]))] if Sc is not None else [])
    base = np.full(len(y), -1)
    pr = [np.full(len(y), -1) for _ in srcs]
    for k in range(5):
        te = fold == k
        fp = fit_probe(X[~te], y[~te])
        base[te] = fp(X[te])
        for n, (Z, c) in enumerate(srcs):
            pr[n][te] = fp(Z["H"][te, c].astype(np.float32))
    P, C = S["P"], S["C"]
    r = dict(model=model, unseen=model in UNSEEN, has_ctrl_g6=Sc is not None)
    m = (P >= 0) & (C >= 0)
    for what, a in (("probe", base), ("answer", ans)):  # depth 0: partner items run in full
        ok_o, ok_s = m & (a[np.maximum(P, 0)] >= 0), m & (a[np.maximum(C, 0)] >= 0)
        r[f"{what}_other@0"] = np.mean(a[P[ok_o]] == y[P[ok_o]])
        r[f"{what}_same@0"] = np.mean(a[C[ok_s]] != y[ok_s])
    for n, (Z, c) in enumerate(srcs):
        name = str(Z["conds"][c])
        tag = f"{100 * depth(name, nL):g}"
        kind = "other" if name.startswith("swap") else "same"
        q = P if kind == "other" else C
        valid = m & (Z["ans"][:, c] >= 0)
        tgt = (y[np.maximum(q, 0)]) if kind == "other" else 1 - y  # binary illusion labels: "other label" = 1 - y
        r[f"probe_{kind}@{tag}"] = np.mean(pr[n][valid] == tgt[valid])
        r[f"answer_{kind}@{tag}"] = np.mean(Z["ans"][valid, c] == tgt[valid])
    for what in ("probe", "answer"):
        for t in ("0", "12.5", "25", "50", "75"):
            if f"{what}_other@{t}" in r and f"{what}_same@{t}" in r:
                r[f"D_{what}@{t}"] = r[f"{what}_other@{t}"] - r[f"{what}_same@{t}"]
    rows.append(r)

df = pd.DataFrame(rows).set_index("model")
os.makedirs(f"{OUT}/g6", exist_ok=True)
df.to_csv(f"{OUT}/g6/swap_delta.csv")
pd.set_option("display.width", 250)
print((df.select_dtypes("number") * 1).round(3).T.to_string())


def verdict(M, label):
    need = int(np.ceil(2 * len(M) / 3))
    med = {t: M[f"D_probe@{t}"].median() for t in ("0", "12.5", "25", "50", "75") if f"D_probe@{t}" in M and M[f"D_probe@{t}"].notna().all()}
    m4 = all(t in med and med[t] >= 0.30 for t in ("0", "12.5", "25"))
    m5 = "75" in med and med["75"] <= 0.10
    k6 = int((M["D_answer@25"] <= M["D_probe@25"] - 0.15).sum())
    print(f"\n[{label}, n={len(M)}] median D_probe {{{', '.join(f'{t}: {v:.3f}' for t, v in med.items())}}}")
    print(f"M4 D_probe >= .30 at 0/12.5/25 -> {'PASS' if m4 else 'FAIL'}; M5 D_probe@75 <= .10 -> {'PASS' if m5 else 'FAIL'}; "
          f"M6 D_answer@25 <= D_probe@25 - .15: {k6}/{len(M)} (need {need}) -> {'PASS' if k6 >= need else 'FAIL'}")


verdict(df, "all G5 models")
if df.unseen.any():
    verdict(df[df.unseen], "unseen models (descriptive)")
