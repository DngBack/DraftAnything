"""VLMBias generation calibration, offline, with 50/50 splits grouped by base image (each image appears at 3 resolutions
x 2 question wordings; gen_calib's split was stratified by (subset, label) only, so copies fell on both halves).
Same calibrator as gen_calib (LR on centred decision-step scores, per subset). Prints ungrouped vs grouped, 20 draws.

usage: python vlmbias_grouped.py   (results/vlmbias_grouped.csv)
"""
import glob, os, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
from datasets import load_dataset
from gen_calib import SUBSETS, ILLUSIONS, OUT

SETS = {"illusions": sorted(ILLUSIONS), "Car Logos": ["Car Logos"], "Dice Pattern": ["Dice Pattern"], "Tally Pattern": ["Tally Pattern"]}
ds = load_dataset("anvo25/vlms-are-biased", split="main")
KEYS = [re.sub(r"_px\d+", "", p) for p, t in zip(ds["image_path"], ds["sub_topic"]) if t in SUBSETS]
X = lambda S: np.stack([np.asarray(s) - np.mean(s) for s in S])


def evaluate(d, tr, groups):
    raw = cal = rem = 0
    okr, okc = [], []
    for g in groups:
        gm = (d.group == g).to_numpy()
        a, b = d[gm & tr], d[gm & ~tr]
        cands = SUBSETS[g]
        pred = b.raw.to_numpy(dtype=object).copy()
        fit = a[a.scores.notna() & a.y.isin(cands)]
        if fit.y.nunique() >= 2:
            clf = LogisticRegression(max_iter=5000).fit(X(fit.scores), fit.y.map(cands.index))
            has = b.scores.notna().to_numpy()
            if has.any():
                pred[has] = [cands[c] for c in clf.predict(X(b.scores[has]))]
        mp, maj = a.groupby("raw").y.agg(lambda v: v.mode()[0]), a.y.mode()[0]
        okr.append((b.raw == b.y).to_numpy()); okc.append(pred == b.y.to_numpy(dtype=object))
        rem += (b.raw.map(lambda r: mp.get(r, maj)) == b.y).sum()
    okr, okc = np.concatenate(okr), np.concatenate(okc)
    return okr.mean(), okc.mean(), rem / len(okr), okr, okc


rows = []
for p in sorted(glob.glob(f"{OUT}/gencal_vlmbias_*.parquet")):
    df = pd.read_parquet(p).assign(key=KEYS)
    model = os.path.basename(p)[15:-8]
    for name, groups in SETS.items():
        d = df[df.group.isin(groups)]
        if d.scores.isna().mean() > 0.10:  # the paper's exclusion rule
            continue
        res = {}
        for mode in ("ungrouped", "grouped"):
            rng, acc = np.random.default_rng(0), []
            for _ in range(20):
                tr = np.zeros(len(d), bool)
                for g in groups:
                    gm = (d.group == g).to_numpy()
                    if mode == "grouped":
                        u = rng.permutation(d.key[gm].unique())
                        tr[gm] = d.key[gm].isin(u[: len(u) // 2]).to_numpy()
                    else:
                        for _, gg in d[gm].groupby("y"):
                            sel = rng.permutation(gg.index)[: len(gg) // 2]
                            tr[d.index.get_indexer(sel)] = True
                r, c, m, okr, okc = evaluate(d, tr, groups)
                n01, n10 = int((okc & ~okr).sum()), int((~okc & okr).sum())
                acc.append((r, c, m, binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0))
            a = np.array(acc)
            res[mode] = a
            rows.append(dict(model=model, set=name, split=mode, raw=100 * a[:, 0].mean(), cal=100 * a[:, 1].mean(),
                             gain=100 * (a[:, 1] - a[:, 0]).mean(), gain_sd=100 * (a[:, 1] - a[:, 0]).std(),
                             remap=100 * a[:, 2].mean(), frac_p05=(a[:, 3] < 0.05).mean()))
out = pd.DataFrame(rows).round(2)
out.to_csv(f"{OUT}/vlmbias_grouped.csv", index=False)
print(out.to_string(index=False))
