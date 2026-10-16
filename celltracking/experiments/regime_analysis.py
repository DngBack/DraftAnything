"""Post hoc: in which regime does family-centre refinement beat a direct matcher? (review item 3.1)

Per window, label-derived regime covariates (evaluator only, never seen by a model):
  n_anchors      ancestors in the window (crowding)
  disp           median |target - true ancestor| after removing the colony's mean shift, in anchor spacings
  mixing         fraction of scored targets whose nearest scored neighbour descends from another ancestor
  generations    log2(n_targets / n_anchors)
joined to existing per-window accuracies of
  in-domain:  full LineageRAFT (raft_v3*, 3 seeds) vs direct matcher (dm_darcy*, 3 seeds), OIAM M3/M4
  zero-shot:  RodSim LineageRAFT (raft_sim_loc*, 3 seeds) vs RodSim direct matcher (dm_sim_loc*_lr, 3 seeds)
The window loops (and detection shuffles, seed 20261002) are exactly those of raft_external_eval.py.

  python experiments/regime_analysis.py covariates   # CPU, writes regime_covariates.parquet
  python experiments/regime_analysis.py summary
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
import experiments.raft_external_eval as ree  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402

OUT = RESULTS / "regime_covariates.parquet"
SEEDS_FULL = ["raft_v3", "raft_v3_s1", "raft_v3_s2"]
SEEDS_DM = ["dm_darcy", "dm_darcy_s1", "dm_darcy_s2"]
SIM_DM = ["dm_sim_loc_lr", "dm_sim_loc_s1_lr", "dm_sim_loc_s2_lr"]


def covariates(e, valid, truth):
    a = e["anchor"][["center_y", "center_x"]].to_numpy(np.float64)
    b = e["target"][["center_y", "center_x"]].to_numpy(np.float64)
    out = {"n_anchors": len(a), "n_targets": len(b), "generations": np.log2(max(len(b), 1) / max(len(a), 1))}
    if len(a) < 2 or len(valid) < 2:
        return {**out, "disp": np.nan, "mixing": np.nan}
    spacing = float(np.median(cKDTree(a).query(a, k=2)[0][:, 1]))
    d = b[valid] - a[truth]
    d = d - d.mean(0)  # a pure translation is free for every method
    _, nn = cKDTree(b[valid]).query(b[valid], k=2)
    return {**out, "disp": float(np.median(np.linalg.norm(d, axis=1)) / max(spacing, 1e-6)),
            "mixing": float(np.mean(truth[nn[:, 1]] != truth))}


class CovRunner:
    def __init__(self, *_, **__):
        self.rows = []

    def window(self, dataset, seq, wid, minutes, e, valid, truth, shape):
        if not len(truth):
            return
        self.rows.append({"dataset": dataset, "sequence": seq, "window_id": wid, "gap_minutes": minutes,
                          **covariates(e, np.asarray(valid), np.asarray(truth))})


def collect():
    ree.Runner = CovRunner
    ree.summary = lambda out: None
    ree.run(type("A", (), {"tags": [], "decoders": ["argmax"], "ensemble": False, "out": "regime_tmp",
                           "datasets": ["OIAM", "recA", "rpsM", "FAST", "LFCT", "T98G"]})())
    tmp = RESULTS / "raft_external_regime_tmp.parquet"
    tmp.rename(OUT)


def margins():
    cov = pd.read_parquet(OUT)
    # in-domain: OIAM-trained full vs direct matcher
    acc = {}
    for name, tags in (("full", SEEDS_FULL), ("dm", SEEDS_DM)):
        ds = [pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet").query("method == @t + '_argmax'")
              .set_index(["sequence", "H", "window_id"]).accuracy for t in tags]
        acc[name] = pd.concat(ds, axis=1).mean(axis=1)
    ind = pd.DataFrame(acc).reset_index().rename(columns={"H": "gap_minutes"})
    ind["regime"], ind["dataset"] = "in-domain", "OIAM"
    # zero-shot: RodSim-trained full vs direct matcher
    v1 = pd.read_parquet(RESULTS / "raft_external_v1.parquet")
    v3 = pd.read_parquet(RESULTS / "raft_external_v3.parquet")
    dm = pd.read_parquet(RESULTS / "raft_external_dm_sim.parquet")
    key = ["dataset", "sequence", "window_id", "gap_minutes"]
    raft = pd.concat([v1[v1.method == "raft_sim_loc_argmax"],
                      v3[v3.method.isin(["raft_sim_loc_s1_argmax", "raft_sim_loc_s2_argmax"])]])
    zs = pd.DataFrame({"full": raft.groupby(key).accuracy.mean(),
                       "dm": dm[dm.method.isin([f"{t}_argmax" for t in SIM_DM])].groupby(key).accuracy.mean(),
                       "nearest": dm[dm.method == "nearest_anchor"].set_index(key).accuracy}).reset_index()
    zs["regime"] = "zero-shot"
    d = pd.concat([ind, zs], ignore_index=True)
    d = d.merge(cov, on=key, how="left", validate="many_to_one")
    d["diff"] = d.full - d.dm
    return d


def summary():
    import statsmodels.formula.api as smf
    d = margins()
    print("missing covariates:", int(d.disp.isna().sum()), "of", len(d))
    d = d.dropna(subset=["disp", "diff"])
    d["crowd"] = pd.cut(d.n_anchors, [0, 25, 100, 400, 1e5], labels=["<=25", "26-100", "101-400", ">400"])
    d["dispbin"] = pd.cut(d.disp, [0, 0.5, 1, 2, 1e3], labels=["<.5", ".5-1", "1-2", ">2"])
    print("\n== covariate medians per dataset/regime ==")
    print(d.groupby(["regime", "dataset"])[["n_anchors", "disp", "mixing", "generations"]].median().round(2)
          .to_string())
    print("\n== full - direct matcher, by crowding ==")
    print(d.groupby(["regime", "dataset", "crowd"], observed=True)["diff"].agg(["mean", "count"]).round(3)
          .unstack("crowd").to_string())
    print("\n== full - direct matcher, by displacement (spacings) ==")
    print(d.groupby(["regime", "dispbin"], observed=True)["diff"].agg(["mean", "count"]).round(3).to_string())
    print("\n== zero-shot only, crowding x displacement (mean diff, count) ==")
    z = d[d.regime == "zero-shot"]
    print(z.groupby(["crowd", "dispbin"], observed=True)["diff"].agg(["mean", "count"]).round(3)
          .unstack("dispbin").to_string())
    for reg in ("in-domain", "zero-shot"):
        s = d[d.regime == reg].assign(lcrowd=lambda x: np.log2(x.n_anchors))
        f = "diff ~ lcrowd + disp + mixing + generations" + (" + C(dataset)" if reg == "zero-shot" else "")
        # in-domain has two test movies only, so clustering by sequence is meaningless there: HC3 instead
        m = (smf.ols(f, s).fit(cov_type="HC3") if reg == "in-domain" else
             smf.ols(f, s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.dataset + s.sequence)[0]}))
        print(f"\n== OLS {reg}: {f} ({'HC3' if reg == 'in-domain' else 'SE clustered by sequence'}; n={len(s)}) ==")
        print(m.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(4).to_string())
    d.drop(columns=["crowd", "dispbin"]).to_parquet(RESULTS / "regime_analysis.parquet", index=False)


if __name__ == "__main__":
    {"covariates": collect, "summary": summary}[sys.argv[1]]()
