"""R1 A + E (docs/r1_protocol.md): readout-capacity ladder, nested layer selection, bootstrap intervals, fold seeds.

Per cell (24 closed-form + 40 G3), on the same folds: D (scores + per-class offset), C (affine on scores, the paper's
calibrator), M (MLP on scores), F (linear on the final-layer answer-position state), P (linear, layer/position chosen
by inner CV on the training folds), PM (MLP on P's layer). Writes results/r1_ladder.csv.

usage: python ladder.py
"""
import glob, os
import numpy as np, pandas as pd, torch
from pilot import OUT, probe_cv
from diag_gen import G3

F_ = torch.nn.functional


def fit(Xtr, ytr, C, kind, steps=300):
    """kind: diag (b only, raw scores), lin, mlp. Returns a predict function."""
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device="cuda"); ytr = torch.as_tensor(ytr, device="cuda")
    if kind == "diag":
        mu, sd = 0.0, 1.0
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    Z = (Xtr - mu) / sd
    d = Z.shape[1]
    if kind == "diag":
        b = torch.zeros(C, device="cuda", requires_grad=True); ps, net = [b], lambda z: z + b
    elif kind == "lin":
        lin = torch.nn.Linear(d, C).cuda(); torch.nn.init.zeros_(lin.weight); ps, net = list(lin.parameters()), lin
    else:
        net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, C)).cuda(); ps = list(net.parameters())
    opt = torch.optim.Adam(ps, lr=1e-2 if kind != "mlp" else 1e-3, weight_decay=1e-3 if kind != "mlp" else 1e-2)
    for _ in range(steps):
        opt.zero_grad(); F_.cross_entropy(net(Z), ytr).backward(); opt.step()
    @torch.no_grad()
    def predict(X):
        return net((torch.as_tensor(X, dtype=torch.float32, device="cuda") - mu) / sd).argmax(-1).cpu().numpy()
    return predict


def folds(n, groups, seed=0, k=5):  # identical to pilot.probe_cv
    g = torch.Generator().manual_seed(seed)
    if groups is None:
        return (torch.randperm(n, generator=g) % k).numpy()
    u, inv = np.unique(groups, return_inverse=True)
    return (torch.randperm(len(u), generator=g) % k).numpy()[inv]


def best_layer_probe(d, y, groups, seed=0):
    """the paper's probe: best (position, layer) on all data."""
    best = None
    for pos in ("last", "imgm"):
        pr, _ = probe_cv(d[pos], y, seed=seed, groups=groups)
        acc = (pr == y).mean(1)
        if best is None or acc.max() > best[0]:
            best = (acc.max(), pr[acc.argmax()])
    return best[1]


def boot(y_err_type, groups, rng, B=1000):
    """95% interval of the share of each type among errors, resampling groups."""
    u, inv = np.unique(groups, return_inverse=True)
    out = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
        out.append([np.sum(w * (y_err_type == t)) / max(np.sum(w * (y_err_type >= 0)), 1) for t in range(3)])
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def cell(path, g3):
    d = np.load(path)
    y, ans, s = d["y"], d["ans"], d["scores"].astype(np.float32)
    groups = d["key"] if g3 else None
    grp = d["group"] if g3 else np.zeros(len(y), int)
    C, n = s.shape[1], len(y)
    fold = folds(n, groups)
    pr = {k: ans.copy() for k in ("D", "C", "M", "F", "P", "PM")}
    for f in range(5):
        tr, te = fold != f, fold == f
        for g in np.unique(grp):  # score readouts per subset (as the paper's calibrator)
            a, b = tr & (grp == g), te & (grp == g)
            if len(np.unique(y[a])) < 2:
                continue
            for k, kind in (("D", "diag"), ("C", "lin"), ("M", "mlp")):
                pr[k][b] = fit(s[a], y[a], C, kind)(s[b])
        H = {pos: d[pos][tr].astype(np.float32) for pos in ("last", "imgm")}
        pr["F"][te] = fit(d["last"][tr, -1], y[tr], C, "lin")(d["last"][te, -1])
        # inner CV over (position, layer) on the training folds only
        best = None
        for pos in ("last", "imgm"):
            ip, _ = probe_cv(H[pos], y[tr], k=4, groups=None if groups is None else groups[tr])
            acc = (ip == y[tr]).mean(1)
            if best is None or acc.max() > best[0]:
                best = (acc.max(), pos, int(acc.argmax()))
        _, pos, li = best
        pr["P"][te] = fit(H[pos][:, li], y[tr], C, "lin")(d[pos][te, li])
        pr["PM"][te] = fit(H[pos][:, li], y[tr], C, "mlp")(d[pos][te, li])
    acc = {k: np.mean(v == y) for k, v in pr.items()}
    m, err = np.mean(ans == y), ans != y
    pb = best_layer_probe(d, y, groups)
    acc["Pbest"] = np.mean(pb == y)
    typ = np.where(~err, -1, np.where(pr["C"] == y, 0, np.where(pr["P"] == y, 1, 2)))  # bias / loss / perc (nested P)
    ne = max(err.sum(), 1)
    loss = typ == 1
    ci = boot(typ, groups if groups is not None else np.arange(n), np.random.default_rng(0))
    rec = lambda c, p: (c - m) / (p - m) if p > m else np.nan
    seeds = [rec(np.mean(probe_cv(s, y, seed=sd, groups=groups)[0] == y), np.mean(best_layer_probe(d, y, groups, sd) == y))
             for sd in range(5)] if not g3 else []
    r = dict(cell=os.path.basename(path).replace("_states.npz", ""), g3=g3, n=n, n_err=int(err.sum()), model=m,
             **{f"acc_{k}": v for k, v in acc.items()},
             biasD=np.sum(err & (pr["D"] == y)) / ne, bias=np.sum(typ == 0) / ne, loss=loss.sum() / ne, perc=np.sum(typ == 2) / ne,
             loss_atF=np.sum(loss & (pr["F"] == y)) / max(loss.sum(), 1),
             bias_lo=ci[0, 0], bias_hi=ci[1, 0], loss_lo=ci[0, 1], loss_hi=ci[1, 1], perc_lo=ci[0, 2], perc_hi=ci[1, 2],
             rec_best=rec(acc["C"], acc["Pbest"]), rec_nested=rec(acc["C"], acc["P"]))
    r.update({f"rec_s{i}": v for i, v in enumerate(seeds)})
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r


def main():
    cf = pd.concat([pd.read_csv(f"{OUT}/g15_table.csv"), pd.read_csv(f"{OUT}/g2_table.csv")]).drop_duplicates("cell")
    cf_cells = set(cf[cf.n_err >= 30].cell)
    g3t = pd.read_csv(f"{OUT}/g3_table.csv")
    g3_cells = set(g3t[(g3t.n_err >= 30) & (g3t.nodec <= 0.10) & (g3t.text < 0.75)].cell)
    paths = [(p, False) for p in sorted(glob.glob(f"{OUT}/*_states.npz")) if os.path.basename(p)[:-11] in cf_cells] + \
            [(p, True) for p in sorted(glob.glob(f"{G3}/*_states.npz")) if os.path.basename(p)[:-11] in g3_cells]
    print(len(paths), "cells", flush=True)
    done = pd.read_csv(f"{OUT}/r1_ladder.csv") if os.path.exists(f"{OUT}/r1_ladder.csv") else pd.DataFrame(columns=["cell", "g3"])
    rows = done.to_dict("records")
    for p, g3 in paths:
        name = os.path.basename(p)[:-11]
        if ((done.cell == name) & (done.g3 == g3)).any():
            continue
        rows.append(cell(p, g3))
        pd.DataFrame(rows).to_csv(f"{OUT}/r1_ladder.csv", index=False)
    report(pd.DataFrame(rows))


def report(df):
    for g3, name in ((False, "closed-form"), (True, "G3")):
        t = df[df.g3 == g3]
        gain = t.acc_C - t.model
        a1 = t[gain >= 0.02]
        a1_hit = ((a1.acc_D - a1.model) >= 0.8 * (a1.acc_C - a1.model))
        print(f"\n== {name}: {len(t)} cells")
        print(f"A1 D recovers >=80% of C's gain: {int(a1_hit.sum())}/{len(a1)} -> {'holds' if a1_hit.mean() >= 2/3 else 'fails'}")
        a2 = (t.acc_M - t.acc_C) <= 0.02
        print(f"A2 M - C <= 2 pts: {int(a2.sum())}/{len(t)} -> {'holds' if a2.mean() >= 2/3 else 'fails'}")
        a3 = (t.acc_PM - t.acc_P) <= 0.02
        print(f"A3 PM - P <= 2 pts: {int(a3.sum())}/{len(t)} -> {'holds' if a3.mean() >= 2/3 else 'fails'}")
        nb, nn = int((t.rec_best >= 0.5).sum()), int((t.rec_nested >= 0.5).sum())
        print(f"A4 Rec>=50%: best-layer {nb} vs nested {nn} -> {'holds' if abs(nb - nn) <= 2 else 'fails'}")
        print(f"   loss errors already readable from the final state (median): {t.loss_atF.median():.2f}")
        if not g3:
            print("   fold seeds: Rec>=50% cells per seed:", [int((t[f"rec_s{i}"] >= .5).sum()) for i in range(5)])


if __name__ == "__main__":
    main()
