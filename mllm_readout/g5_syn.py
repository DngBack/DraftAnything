"""G5 S (docs/g5_protocol.md): shortcut-controlled synthetic illusions, tests S1-S4.

Readouts: pixel baseline (64 statistics, illusyn.pixel_stats), SigLIP2 / DINOv2 grid (explore_input), native l0 / pre
(explore_native), decision state at 75% (G3 states in results/g5_syn), calibrator on candidate logits. Cross-regime:
train on regime A, test on B and vice versa (mean). In-regime: 5-fold CV inside each regime (mean over regimes).
Output: results/g5/syn_<set>.csv.   usage: python g5_syn.py [illusyn | illusynL]
"""
import glob, os, sys, numpy as np, pandas as pd
from pilot import probe_cv, OUT
from steer_g3 import fit_probe
from illusyn import items, pixel_stats

G5SET = ["qwen3vl4b", "qwen3vl8b", "qwen35_4b", "qwen35_9b", "gemma3_4b", "internvl35_8b", "llava15_7b", "qwen2vl_7b"]
INP, NAT = f"{OUT}/explore_input", f"{OUT}/explore_native"


def clean(X):
    return np.clip(np.nan_to_num(X.astype(np.float32), posinf=0, neginf=0), -1e4, 1e4)


def fit_mlp(X, y, hidden=256, steps=500, wd=1e-2, seed=0):
    """Addendum 3 (b): two-layer MLP probe, same optimiser family as fit_probe"""
    import torch
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32, device="cuda"); y = torch.as_tensor(y, device="cuda")
    mu, sd = X.mean(0), X.std(0) + 1e-4
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, int(y.max()) + 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.cross_entropy(net((X - mu) / sd), y).backward(); opt.step()
    return lambda Z: net((torch.as_tensor(Z, dtype=torch.float32, device="cuda") - mu) / sd).argmax(-1).detach().cpu().numpy()


def cross(X, y, reg, fit=fit_probe):
    a = [np.mean(fit(X[reg == s], y[reg == s])(X[reg != s]) == y[reg != s]) for s in "AB"]
    return float(np.mean(a))


def within(X, y, reg, fit=None):
    if fit is None:
        return float(np.mean([np.mean(probe_cv(X[reg == s], y[reg == s])[0] == y[reg == s]) for s in "AB"]))
    acc = []
    for s in "AB":
        Xs, ys = X[reg == s], y[reg == s]
        fold = np.random.default_rng(0).permutation(len(ys)) % 5
        acc += [np.mean(fit(Xs[fold != f], ys[fold != f])(Xs[fold == f]) == ys[fold == f]) for f in range(5)]
    return float(np.mean(acc))


def diagnostics(name, X, y, reg, lon, diff):
    """Addendum 3: (a) signed task on unequal items, (b) MLP probe for equal/unequal, (c) linear cross-regime by |diff|"""
    u = lon >= 0
    r = {f"{name}_signed_within": within(X[u], lon[u], reg[u]), f"{name}_signed_cross": cross(X[u], lon[u], reg[u]),
         f"{name}_mlp_within": within(X, y, reg, fit_mlp), f"{name}_mlp_cross": cross(X, y, reg, fit_mlp)}
    for s in "AB":  # (c): train on regime s, accuracy on the other regime per |diff| (equal items included in each bin)
        p = fit_probe(X[reg == s], y[reg == s])(X[reg != s])
        dd, yy = diff[reg != s], y[reg != s]
        for v in sorted(set(diff[diff > 0])):
            m = (dd == v) | (dd == 0)
            r[f"{name}_cross_d{v:g}"] = r.get(f"{name}_cross_d{v:g}", 0) + np.mean(p[m] == yy[m]) / 2
    return r


SET = sys.argv[1] if len(sys.argv) > 1 else "illusyn"
SYN = f"{OUT}/g6_syn" if SET == "illusynC" else f"{OUT}/g5_syn"  # G6 S': fresh stimuli (docs/g6_protocol.md)
its = {"illusyn": lambda: items(), "illusynL": lambda: items(diffs=(0.30, 0.40, 0.50), seed=2027, tag="L_"),
       "illusynC": lambda: items(diffs=(0.30, 0.40, 0.50), seed=2028, tag="C_")}[SET]()
pix_all = np.stack([pixel_stats(it["image"]) for it in its])
reg_all = np.array([it["regime"] for it in its])
fam_all = np.array([it["group"] for it in its])
y_all = np.array([int(it["y"] == "No") for it in its])
lon_all = np.array([it["longer"] for it in its])
diff_all = np.array([it["diff"] for it in its])
ext_all = {k: clean(np.load(f"{INP}/{k}_{SET}.npz")["feats"][:, 0].reshape(len(its), -1)) for k in ("siglip2", "dinov2")}
glob_rows = dict(model="(model-free)", pixel_cross=cross(pix_all, y_all, reg_all), pixel_within=within(pix_all, y_all, reg_all),
                 **{f"{k}_cross": cross(X, y_all, reg_all) for k, X in ext_all.items()},
                 **{f"{k}_within": within(X, y_all, reg_all) for k, X in ext_all.items()})
for f in ("mullerlyer", "ponzo", "verticalhorizontal"):  # reported: per family
    m = fam_all == f
    glob_rows[f"siglip2_cross_{f}"] = cross(ext_all["siglip2"][m], y_all[m], reg_all[m])
    glob_rows[f"pixel_cross_{f}"] = cross(pix_all[m], y_all[m], reg_all[m])
for k, X in [("pixel", pix_all)] + list(ext_all.items()):
    glob_rows.update(diagnostics(k, X, y_all, reg_all, lon_all, diff_all))
print({k: round(v, 3) for k, v in glob_rows.items() if k != "model"}, flush=True)

rows = []
for st in sorted(glob.glob(f"{SYN}/*_{SET}_states.npz")):
    model = os.path.basename(st)[: -len(f"_{SET}_states.npz")]
    d = np.load(st, allow_pickle=True)
    idx, ans = d["idx"], d["ans"]
    y = np.array([d["y"][k] for k in range(len(idx))])  # candidate index: 0 = Yes (equal), 1 = No
    reg, fam = reg_all[idx], fam_all[idx]
    nL = d["last"].shape[1] - 1
    L = round(0.75 * nL)
    H = clean(d["last"][:, L + 1])
    sc = np.asarray(d["scores"], np.float32)
    groups = np.array([str(k) for k in d["key"]])
    r = dict(model=model, n=len(y), raw=np.mean(ans == y), raw_A=np.mean((ans == y)[reg == "A"]), raw_B=np.mean((ans == y)[reg == "B"]),
             say_no=np.mean(ans == 1), dec_cross=cross(H, y, reg), dec_within=within(H, y, reg))
    pc = probe_cv(sc - sc.mean(1, keepdims=True), y, groups=groups)[0]
    pd_ = probe_cv(H, y, groups=groups)[0]
    nf = f"{NAT}/{model}_{SET}.npz"
    if os.path.exists(nf):
        N = np.load(nf)
        l0, pre = clean(N["l0"][idx]), clean(N["pre"][idx])
        r.update(l0_cross=cross(l0, y, reg), l0_within=within(l0, y, reg), pre_cross=cross(pre, y, reg), pre_within=within(pre, y, reg))
        r.update(diagnostics("l0", l0, y, reg, lon_all[idx], diff_all[idx]))
        r.update(diagnostics("dec", H, y, reg, lon_all[idx], diff_all[idx]))
        pe = probe_cv(l0, y, groups=groups)[0]
        err, C, D, E = ans != y, pc == y, pd_ == y, pe == y
        r.update(cal=np.mean(C), dec=np.mean(D), l0=np.mean(E), errors=int(err.sum()), BIAS=int((err & C).sum()),
                 UNSPOKEN=int((err & ~C & D & ~E).sum()), INHERITED=int((err & ~C & D & E).sum()),
                 DROPPED=int((err & ~C & ~D & E).sum()), UNSEEN=int((err & ~C & ~D & ~E).sum()))
    rows.append(r)
    print(model, {k: round(float(v), 3) if isinstance(v, float) else v for k, v in r.items() if k != "model"}, flush=True)

R = pd.DataFrame([glob_rows] + rows)
os.makedirs(f"{OUT}/g5", exist_ok=True)
R.to_csv(f"{OUT}/g5/syn_{SET}.csv", index=False)
M = pd.DataFrame(rows)
if M.empty:
    raise SystemExit
M = M[M.model.isin(G5SET)].set_index("model")
need = int(np.ceil(2 * len(M) / 3))
s1 = glob_rows["siglip2_cross"] >= 0.70 and glob_rows["pixel_cross"] <= 0.60
print(f"\nS1 SigLIP2 cross {glob_rows['siglip2_cross']:.3f} (>= .70), pixel cross {glob_rows['pixel_cross']:.3f} (<= .60) -> {'PASS' if s1 else 'FAIL'}")
if "l0_cross" in M:
    M = M.dropna(subset=["l0_cross"])
    need = int(np.ceil(2 * len(M) / 3))
    k2 = int((M.l0_cross >= 0.65).sum())
    k3 = int((M.raw <= M.l0_within - 0.15).sum())
    ok = M[M.errors >= 30]
    k4 = int((ok.INHERITED > ok.UNSPOKEN).sum())
    print(f"S2 native l0 cross >= .65: {k2}/{len(M)} (need {need}) -> {'PASS' if k2 >= need else 'FAIL'}")
    print(f"S3 model <= l0 within - 15: {k3}/{len(M)} (need {need}) -> {'PASS' if k3 >= need else 'FAIL'}")
    print(f"S4 INHERITED > UNSPOKEN: {k4}/{len(ok)} cells with >= 30 errors (need {int(np.ceil(2 * len(ok) / 3))}) -> "
          f"{'PASS' if k4 >= np.ceil(2 * len(ok) / 3) else 'FAIL'}")
if SET == "illusynC" and "l0_cross" in M:  # G6 S5-S8
    k5 = int(((M.l0_within >= 0.65) & (M.l0_cross <= M.l0_within - 0.10)).sum())
    k6 = int((M.raw <= M.l0_within - 0.10).sum())
    k7 = int((M.l0_signed_cross >= M.l0_cross + 0.05).sum())
    s8 = glob_rows["siglip2_within"] >= 0.70 and glob_rows["siglip2_cross"] <= 0.65 and glob_rows["pixel_within"] <= 0.62
    for t, k in (("S5 l0 within >= .65 and across <= within - .10", k5), ("S6 model <= l0 within - .10", k6),
                 ("S7 l0 signed across >= l0 across + .05", k7)):
        print(f"{t}: {k}/{len(M)} (need {need}) -> {'PASS' if k >= need else 'FAIL'}")
    print(f"S8 SigLIP2 within {glob_rows['siglip2_within']:.3f} (>= .70), across {glob_rows['siglip2_cross']:.3f} (<= .65), "
          f"pixel within {glob_rows['pixel_within']:.3f} (<= .62) -> {'PASS' if s8 else 'FAIL'}")
pd.set_option("display.width", 250)
print((M.select_dtypes("number") * 1).round(3).to_string())
