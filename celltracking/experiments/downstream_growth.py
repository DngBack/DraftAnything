"""Downstream biology from two snapshots: per-ancestor family size (clonal growth).

For every ancestor i of a window, its family size n_i is the number of targets assigned
to it. We compare what each method recovers with the ground truth on readouts a
biologist would compute from lineage:
  - exact family size per ancestor, correlation of n_hat with n,
  - growth heterogeneity (CV of family size across ancestors),
  - size control: Spearman correlation of ancestor length with family size
    (longer cells divide sooner), and the division probability of the longest vs
    shortest third of ancestors.
OIAM M3/M4, GT tracks for scoring.

  python experiments/downstream_growth.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_glue import Movie  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402

TAGS = ("raft_v3", "raft_pseudo", "raft_sim_loc")
HS = (33, 66, 132)


def ancestors_df(movie, window_id, s, H, fa, name, assign, ok):
    n = np.bincount(assign[ok], minlength=len(fa["pos"]))
    return pd.DataFrame({"movie": movie, "window": window_id, "H": H, "start": s, "method": name,
                         "ancestor": np.arange(len(n)), "length": np.sqrt(12) * fa["major"], "n": n})


def main():
    models = {t: load(t) for t in TAGS}
    rng = np.random.default_rng(1)
    rows = []
    for alias in ("M3", "M4"):
        m = Movie(alias)
        for H in HS:
            starts = [k for k in sorted(m.frames) if k + H in m.frames and len(m.frames[k]["pos"]) >= 2]
            for w, s in enumerate(rng.choice(starts, 32, replace=False)):
                s = int(s)
                fa, fb, label = m.window(s, s + H)
                ok = label >= 0
                preds = {"gt": label,
                         "nearest": cKDTree(fa["pos"]).query(fb["pos"])[1]}
                for t, (model, margs) in models.items():
                    prob = predict(model, fa, fb, H, m.image_shape, margs.get("no_shape", False),
                                   init=margs.get("init", "radial"))[0]
                    preds[t] = prob[:, :-1].argmax(1)
                for name, p in preds.items():
                    rows.append(ancestors_df(alias, w, s, H, fa, name, p, ok))
            print(alias, H, flush=True)
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(RESULTS / "downstream_growth.parquet", index=False)
    summary(d)


def summary(d=None):
    d = pd.read_parquet(RESULTS / "downstream_growth.parquet") if d is None else d
    gt = d[d.method == "gt"].set_index(["movie", "window", "H", "ancestor"]).n
    out = []
    for (H, name), g in d.groupby(["H", "method"]):
        g = g.set_index(["movie", "window", "H", "ancestor"])
        n, t = g.n, gt.loc[g.index]
        # size control within each window (rank within window to remove time trends)
        rl = g.groupby(level=["movie", "window"]).length.rank(pct=True)
        divided = (n >= 2).astype(float)
        cv = g.groupby(level=["movie", "window"]).n.agg(lambda x: x.std() / x.mean()).mean()
        out.append({"H": H, "method": name,
                    "exact": float(np.mean(n == t)),
                    "pearson": pearsonr(n, t)[0],
                    "cv": cv,
                    "rho_len": spearmanr(rl, n)[0],
                    "p_div_long": divided[rl > 2 / 3].mean(),
                    "p_div_short": divided[rl <= 1 / 3].mean()})
    r = pd.DataFrame(out).set_index(["H", "method"]).round(3)
    print(r.to_string())
    return r


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary()
    else:
        main()
