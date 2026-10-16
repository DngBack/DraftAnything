"""Do larger simulated colonies cover the crowded, large-displacement regime of OIAM? (Direction 2, descriptive)

Windows sampled exactly as in training (lineage_glue.sample, h in [8, 240]); regime covariates of
regime_analysis.covariates. Compares OIAM M0-M2 with RodSim directories given on the command line.

  python experiments/sim_regime_probe.py outputs/sim/v1 outputs/sim/v3_probe
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.colony_sim import SimMovie  # noqa: E402
from experiments.lineage_glue import Movie, sample  # noqa: E402
from experiments.regime_analysis import covariates  # noqa: E402


def regime(movies, n=1500, seed=0):
    rng, rows = np.random.default_rng(seed), []
    for _ in range(n):
        _, fa, fb, label, h = sample(movies, rng, (8, 240))
        ok = np.flatnonzero(label >= 0)
        e = {"anchor": pd.DataFrame(fa["pos"], columns=["center_y", "center_x"]),
             "target": pd.DataFrame(fb["pos"], columns=["center_y", "center_x"])}
        rows.append({"h": h, **covariates(e, ok, label[ok])})
    return pd.DataFrame(rows).dropna()


def main(dirs):
    srcs = {"OIAM M0-M2": [Movie(a) for a in ["M0", "M1", "M2"]]}
    srcs.update({d: [SimMovie(q) for q in sorted(Path(d).glob("colony_*.pkl"))[:64]] for d in dirs})
    out = []
    for name, movies in srcs.items():
        d = regime(movies).assign(source=name)
        big = d[d.n_anchors > 400]
        out.append({"source": name, "colonies": len(movies), "share>400": (d.n_anchors > 400).mean(),
                    "disp50>400": big.disp.median(), "disp90>400": big.disp.quantile(.9),
                    "share>100&disp>2": ((d.n_anchors > 100) & (d.disp > 2)).mean(),
                    "anchors50": d.n_anchors.median(), "anchors90": d.n_anchors.quantile(.9)})
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
