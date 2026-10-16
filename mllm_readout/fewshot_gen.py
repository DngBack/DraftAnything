"""Few-label calibration of free-form generation, offline on the decision-step scores saved by gen_calib.py:
K labelled items per class (from the train half) fit the calibrator; evaluated on the test half; 50 draws."""
import glob, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from gen_calib import CANDS_OF, ILLUSIONS
from pilot import OUT

SETS = {"countbench": ["countbench"], "MMVP": ["MMVP"], "illusions": sorted(ILLUSIONS)}


def run(Ks=(1, 2, 5, 10), draws=50):
    rows = []
    for p in sorted(glob.glob(f"{OUT}/gencal_*.parquet")):
        df, model = pd.read_parquet(p), os.path.basename(p)[7:-8].split("_", 1)[1]
        for name, groups in SETS.items():
            d = df[df.group.isin(groups)]
            if d.empty:
                continue
            te_all = d[~d.train]
            r = dict(model=model, set=name, raw=100 * (te_all.raw == te_all.y).mean())
            rng = np.random.default_rng(0)
            for K in Ks:
                acc = []
                for _ in range(draws):
                    hit = 0
                    for g in groups:
                        cands = CANDS_OF(g)
                        tr = d[d.train & (d.group == g) & d.scores.notna() & d.y.isin(cands)]
                        te = d[~d.train & (d.group == g)]
                        pick = np.concatenate([rng.choice(ix, min(K, len(ix)), replace=False) for _, ix in tr.groupby("y").groups.items()])
                        fit = tr.loc[pick]
                        pred = te.raw.to_numpy(dtype=object).copy()
                        if fit.y.nunique() >= 2:
                            X = lambda S: np.stack([np.asarray(s) - np.mean(s) for s in S])
                            clf = LogisticRegression(max_iter=5000).fit(X(fit.scores), fit.y.map(cands.index))
                            has = te.scores.notna().to_numpy()
                            pred[has] = [cands[c] for c in clf.predict(X(te.scores[has]))]
                        hit += (pred == te.y.to_numpy(dtype=object)).sum()
                    acc.append(100 * hit / len(te_all))
                r[f"K={K}"] = f"{np.mean(acc):.1f}±{np.std(acc):.1f}"
            r["all (50%)"] = 100 * (te_all.cal_offline == te_all.y).mean()
            rows.append(r)
    out = pd.DataFrame(rows).round(1)
    print(out.to_string(index=False))
    out.to_csv(f"{OUT}/fewshot_gen.csv", index=False)


if __name__ == "__main__":
    run()
