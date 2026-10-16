"""Few-shot / label-free calibration of the model's own answer logits (all *_states.npz cells).

batch: label-free, subtract the per-class mean log-prob over the unlabeled test set (Batch Calibration).
k=K:   affine refit (logistic regression on centred candidate logits) from K labelled items per class, 20 draws, test on the rest.
"""
import glob, os, warnings
import numpy as np, pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(ks=(2, 5, 10, 20), draws=20):
    rng = np.random.default_rng(0)
    rows = []
    for p in sorted(glob.glob(f"{OUT}/*_states.npz")):
        d = np.load(p)
        y, s = d["y"], d["scores"].astype(np.float64)
        s = s - s.mean(1, keepdims=True)
        lp = log_softmax(s, 1)
        r = dict(cell=os.path.basename(p)[:-11], n=len(y), model=(d["ans"] == y).mean(),
                 batch=((lp - lp.mean(0)).argmax(1) == y).mean())
        for k in ks:
            acc = []
            for _ in range(draws):
                tr = np.concatenate([rng.choice(np.where(y == u)[0], min(k, (y == u).sum()), replace=False) for u in np.unique(y)])
                te = np.setdiff1d(np.arange(len(y)), tr)
                acc.append((LogisticRegression(max_iter=5000).fit(s[tr], y[tr]).predict(s[te]) == y[te]).mean())
            r[f"k{k}"] = np.mean(acc)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/fewshot_calib.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
