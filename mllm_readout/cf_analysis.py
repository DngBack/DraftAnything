"""R1 B + C (docs/r1_protocol.md): counterfactual partners, prompt transfer, label-prior shift.

B: probe (last position, layer by inner CV on training folds) and calibrator are trained on original items; on each
   original error type we ask whether the probe / calibrator / model is right on the item's counterfactual partner.
C1: calibrator trained on the original prompt, applied to a paraphrased prompt, vs. one fitted on the paraphrase.
C2: calibrator trained on a skewed resample (one class x3), tested class-balanced.
Writes results/r1_cf.csv and results/r1_prior.csv.

usage: python cf_analysis.py
"""
import glob, os
import numpy as np, pandas as pd
from pilot import OUT, probe_cv
from ladder import fit, folds


def cell_cf(path):
    tag = os.path.basename(path).replace("_cf.npz", "")
    c, d = np.load(path), np.load(f"{OUT}/{tag}_states.npz")
    n = len(c["q_y"])  # seeds 0..n-1, the first n items of the original run
    y, ans, s, H = d["y"][:n], d["ans"][:n], d["scores"][:n].astype(np.float32), d["last"][:n]
    assert (c["q_y"] == y).all()
    K = s.shape[1]
    part = -np.ones(n, int); part[c["pidx"]] = np.arange(len(c["pidx"]))
    fold = folds(n, None)
    cal, prb, p_cal, p_prb = ans.copy(), ans.copy(), -np.ones(n, int), -np.ones(n, int)
    q_tr, q_in = c["q_ans"].copy(), c["q_ans"].copy()
    qs = c["q_scores"].astype(np.float32)
    for f in range(5):
        tr, te = fold != f, fold == f
        fc = fit(s[tr], y[tr], K, "lin")
        cal[te] = fc(s[te]); q_tr[te] = fc(qs[te])  # C1: transferred to the paraphrase
        q_in[te] = fit(qs[tr], y[tr], K, "lin")(qs[te])  # C1: fitted on the paraphrase
        ip, _ = probe_cv(H[tr].astype(np.float32), y[tr], k=4)
        li = int((ip == y[tr]).mean(1).argmax())
        fp = fit(H[tr, li].astype(np.float32), y[tr], K, "lin")
        prb[te] = fp(H[te, li])
        j = np.where(te & (part >= 0))[0]
        p_cal[j] = fc(c["p_scores"][part[j]].astype(np.float32))
        p_prb[j] = fp(c["p_last"][part[j], li].astype(np.float32))
    err = ans != y
    typ = np.where(~err, -1, np.where(cal == y, 0, np.where(prb == y, 1, 2)))
    has = part >= 0
    py, pans = np.full(n, -1), np.full(n, -1)
    py[has], pans[has] = c["p_y"][part[has]], c["p_ans"][part[has]]
    r = dict(cell=tag, n=n, n_pairs=int(has.sum()))
    for t, name in enumerate(("bias", "loss", "perc")):
        k = has & (typ == t)
        r[f"n_{name}"] = int(k.sum())
        for who, pred in (("probe", p_prb), ("cal", p_cal), ("model", pans)):
            r[f"{name}_{who}_ok"] = np.mean(pred[k] == py[k]) if k.any() else np.nan
    k = has & ~err
    r["correct_probe_ok"], r["correct_model_ok"] = np.mean(p_prb[k] == py[k]), np.mean(pans[k] == py[k])
    r["probe_flips"] = np.mean(p_prb[has] != prb[has])  # the probe's answer changes with the image
    r.update(para_raw=np.mean(c["q_ans"] == y), para_in=np.mean(q_in == y), para_transfer=np.mean(q_tr == y),
             orig_raw=np.mean(ans == y), orig_cal=np.mean(cal == y))
    return r


def cell_prior(path, rng):
    d = np.load(path)
    y, ans, s = d["y"], d["ans"], d["scores"].astype(np.float32)
    K = s.shape[1]
    fold = folds(len(y), None)
    out = dict(cell=os.path.basename(path).replace("_states.npz", ""))
    acc = {"raw": [], "plain": [], "skewed": []}
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        cls = rng.choice(np.unique(y[tr]))
        sk = np.concatenate([tr] + [tr[y[tr] == cls]] * 2)  # one class x3
        w = 1 / np.bincount(y[te], minlength=K)[y[te]]  # class-balanced test
        for name, pred in (("raw", ans[te]), ("plain", fit(s[tr], y[tr], K, "lin")(s[te])),
                           ("skewed", fit(s[sk], y[sk], K, "lin")(s[te]))):
            acc[name].append(np.sum(w * (pred == y[te])) / w.sum())
    out.update({k: np.mean(v) for k, v in acc.items()})
    return out


def main():
    cf = pd.DataFrame([cell_cf(p) for p in sorted(glob.glob(f"{OUT}/r1/*_cf.npz"))])
    cf.to_csv(f"{OUT}/r1_cf.csv", index=False)
    print(cf.round(3).to_string(index=False))
    b = cf[cf.n_loss >= 20]
    print(f"\nB1 loss errors, probe right on partner (median over {len(b)} cells with >=20 loss errors): "
          f"{b.loss_probe_ok.median():.2f} -> {'holds' if b.loss_probe_ok.median() >= 0.7 else 'fails'}")
    hit = b.loss_model_ok < b.loss_probe_ok
    print(f"B2 model right on partner less often than probe: {int(hit.sum())}/{len(b)} -> {'holds' if len(b) and hit.mean() >= 2/3 else 'fails'}")
    g = cf[(cf.para_in - cf.para_raw) >= 0.02]
    keep = (g.para_transfer - g.para_raw) >= 0.5 * (g.para_in - g.para_raw)
    print(f"C1 transfer keeps >=50% of the in-template gain: {int(keep.sum())}/{len(g)} -> {'holds' if len(g) and keep.mean() >= 2/3 else 'fails'}")
    t = pd.concat([pd.read_csv(f"{OUT}/g15_table.csv"), pd.read_csv(f"{OUT}/g2_table.csv")]).drop_duplicates("cell")
    cells = set(t[t.n_err >= 30].cell)
    rng = np.random.default_rng(0)
    pr = pd.DataFrame([cell_prior(p, rng) for p in sorted(glob.glob(f"{OUT}/*_states.npz")) if os.path.basename(p)[:-11] in cells])
    pr.to_csv(f"{OUT}/r1_prior.csv", index=False)
    gain = pr.plain - pr.raw
    print(f"\nC2 balanced test: median gain plain {100 * gain.median():.1f}, skewed-train {100 * (pr.skewed - pr.raw).median():.1f}; "
          f"retained (median over cells with gain >= 2): {((pr.skewed - pr.raw) / gain)[gain >= 0.02].median():.2f}")


if __name__ == "__main__":
    main()
