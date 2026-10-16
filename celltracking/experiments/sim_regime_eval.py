"""Post hoc: does the moving centre beat the direct matcher *inside* RodSim, and in which regime?

Held-out RodSim v1 colonies (seeds 90000+, never trained on), windows sampled as in training (log-uniform gap
8-240 min). RodSim-trained LineageRAFT (raft_sim_loc*) vs RodSim direct matcher (dm_sim_loc*_lr), positions only,
argmax, three seeds each; covariates as in regime_analysis.py.

  python experiments/sim_regime_eval.py
  python experiments/sim_regime_eval.py v3   # RodSim v3 (docs/sim_v3_protocol.md): v3 held-out, v3-trained arms
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.colony_sim import SimMovie  # noqa: E402
from experiments.family_set import RESULTS, ROOT  # noqa: E402
from experiments.lineage_glue import sample  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.regime_analysis import covariates  # noqa: E402

TAGS = {"full": ["raft_sim_loc", "raft_sim_loc_s1", "raft_sim_loc_s2"],
        "dm": ["dm_sim_loc_lr", "dm_sim_loc_s1_lr", "dm_sim_loc_s2_lr"]}
N_WIN = 600
VARIANTS = {"v1": ("v1_heldout", TAGS, "sim_regime_eval"),
            "v3": ("v3_heldout", {"full": ["raft_simv3", "raft_simv3_s1", "raft_simv3_s2"],
                                  "dm": ["dm_simv3", "dm_simv3_s1", "dm_simv3_s2"]}, "sim_regime_eval_v3")}


def main(variant="v1"):
    held, tags, out = VARIANTS[variant]
    movies = [SimMovie(q) for q in sorted((ROOT / "outputs/sim" / held).glob("colony_*.pkl"))]
    models = {k: [load(t) for t in v] for k, v in tags.items()}
    rng = np.random.default_rng(1)
    rows = []
    for w in range(N_WIN):
        m, fa, fb, label, h = sample(movies, rng, (8, 240))
        ok = np.flatnonzero(label >= 0)
        if len(fa["pos"]) < 2 or len(ok) < 2:
            continue
        e = {"anchor": pd.DataFrame(fa["pos"], columns=["center_y", "center_x"]),
             "target": pd.DataFrame(fb["pos"], columns=["center_y", "center_x"])}
        r = {"window": w, "h": h, **covariates(e, ok, label[ok])}
        for k, ms in models.items():
            acc = []
            for model, margs in ms:
                with torch.no_grad():
                    prob, _, _ = predict(model, fa, fb, h, m.image_shape, no_shape=True,
                                         init=margs.get("init", "radial"))
                acc.append(float(np.mean(prob[:, :-1].argmax(1)[ok] == label[ok])))
            r[k] = float(np.mean(acc))
        rows.append(r)
        if w % 100 == 0:
            print(w, flush=True)
    d = pd.DataFrame(rows)
    d["diff"] = d.full - d.dm
    d.to_parquet(RESULTS / f"{out}.parquet", index=False)
    d["crowd"] = pd.cut(d.n_anchors, [0, 25, 100, 400, 1e5])
    d["dispbin"] = pd.cut(d.disp, [0, 0.5, 1, 2, 4, 1e3])
    print("overall full/dm:", d.full.mean().round(3), d.dm.mean().round(3))
    print(d.groupby("crowd", observed=True)[["full", "dm", "diff"]].mean().round(3).assign(
        n=d.groupby("crowd", observed=True).size()).to_string())
    print(d.groupby(["crowd", "dispbin"], observed=True)["diff"].agg(["mean", "count"]).round(3)
          .unstack("dispbin").to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
