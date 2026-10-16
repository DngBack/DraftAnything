"""R2 E (docs/r2_protocol.md): real-image counterfactual partners on What'sUp Controlled_Images.

Each object pair is photographed in 4 relations with an identical prompt; an item's partners are the other images of
its group. Folds are grouped by object pair, so the probe / calibrator never saw the pair, and partners share the
test fold. For each error type we ask how often the probe / calibrator / model is right on the partners.
Writes results/r2_whatsup.csv.

usage: python whatsup_pairs.py
"""
import glob, os
import numpy as np, pandas as pd
from pilot import OUT, WHATSUP_REL, probe_cv
from ladder import fit, folds


def groups():
    from datasets import load_dataset
    keys = []
    for sp in ("Controlled_Images_A", "Controlled_Images_B"):  # same order as pilot.whatsup_items
        for r in load_dataset("ServiceNow/whatsup_all", split=sp).select_columns(["caption_options"]):
            c = r["caption_options"][0]
            k = next(k for k in WHATSUP_REL if k in c)
            keys.append(f"{sp[-1]}|" + "|".join(c[2:].split(k)))
    return np.array(keys)


def cell(path, G):
    d = np.load(path)
    y, ans, s, H = d["y"], d["ans"], d["scores"].astype(np.float32), d["last"]
    n, K = len(y), s.shape[1]
    g = G[:n]
    # keep complete groups only (Qwen3-VL-32B was run on the first 600 items)
    full = pd.Series(g).map(pd.Series(g).value_counts()).values == 4
    fold = folds(n, g)
    cal, prb = ans.copy(), ans.copy()
    for f in range(5):
        tr, te = fold != f, fold == f
        cal[te] = fit(s[tr], y[tr], K, "lin")(s[te])
        ip, _ = probe_cv(H[tr].astype(np.float32), y[tr], k=4, groups=g[tr])
        li = int((ip == y[tr]).mean(1).argmax())
        prb[te] = fit(H[tr, li].astype(np.float32), y[tr], K, "lin")(H[te, li].astype(np.float32))
    err = ans != y
    typ = np.where(~err, -1, np.where(cal == y, 0, np.where(prb == y, 1, 2)))
    # (item, partner) pairs within complete groups
    idx = pd.DataFrame(dict(i=np.arange(n), g=g))[full]
    pr = idx.merge(idx, on="g", suffixes=("", "_p"))
    pr = pr[pr.i != pr.i_p]
    i, j = pr.i.values, pr.i_p.values
    tag = os.path.basename(path).replace("_states.npz", "")
    r = dict(cell=tag, n=n, n_err=int(err.sum()), model=np.mean(ans == y), calib=np.mean(cal == y), probe=np.mean(prb == y))
    for t, name in ((-1, "correct"), (0, "bias"), (1, "loss"), (2, "perc")):
        k = typ[i] == t
        r[f"n_{name}"] = int((typ[full] == t).sum())
        for who, pred in (("probe", prb), ("cal", cal), ("model", ans)):
            r[f"{name}_{who}_ok"] = np.mean(pred[j[k]] == y[j[k]]) if k.any() else np.nan
    return r


def main():
    G = groups()
    df = pd.DataFrame([cell(p, G) for p in sorted(glob.glob(f"{OUT}/*_spatial_real_states.npz"))])
    df.to_csv(f"{OUT}/r2_whatsup.csv", index=False)
    print(df.round(3).to_string(index=False))
    b = df[df.n_loss >= 20]
    if len(b) < 3:
        print(f"\nE1/E2 underpowered: {len(b)} cells with >= 20 loss errors")
    med = b.loss_probe_ok.median()
    print(f"\nE1 loss errors, probe right on partners (median over {len(b)} cells): {med:.2f} -> {'holds' if med >= .7 else 'fails'}")
    hit = b.loss_model_ok < b.loss_probe_ok
    print(f"E2 model right on partners less often than probe: {int(hit.sum())}/{len(b)} -> {'holds' if len(b) and hit.mean() >= 2/3 else 'fails'}")


if __name__ == "__main__":
    main()
