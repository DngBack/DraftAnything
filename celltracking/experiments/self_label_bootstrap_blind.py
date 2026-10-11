"""Protocol 6 (docs/selflabel_bootstrap_blind_protocol.md): temporal bootstrapping on the blind sources.

Protocol 3 with one-frame links from raft_sim_loc itself instead of Hungarian links.

  python experiments/self_label_bootstrap_blind.py train
  python experiments/self_label_bootstrap_blind.py eval
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_raft import load  # noqa: E402
from experiments.self_label_adapt import (SOURCES, BlindPseudoMovie, all_movies, evaluate, folds,  # noqa: E402
                                          train_movies)
from experiments.self_label_adapt_ext import model_links  # noqa: E402

OUT = "self_label_adapt_blind_boot"


def train(fmap):
    sim = load("raft_sim_loc")[0]
    links = {}
    for source in SOURCES:
        for fold in (0, 1):
            movies = []
            for s, name, mpf, frames, _, _ in all_movies():
                if s != source or fmap[(s, name)] == fold:
                    continue
                if name not in links:
                    links[name] = model_links(frames, mpf, sim)
                movies.append(BlindPseudoMovie(frames, mpf, links[name]))
            train_movies(f"adaptm_{source}_{fold}", movies)


def summary(d):
    seq = d.groupby(["dataset", "sequence", "gap_minutes", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    geo = [c for c in seq if any(g in c for g in ("nearest_anchor", "centroid_translation",
                                                  "radial_expansion", "radial_sector"))]
    src = seq.groupby("dataset").mean()
    cols = ["nearest_anchor", "sim", "sv_sim", "adapt", "sv_adapt", "adaptm", "col_adaptm", "sv_adaptm"]
    print(src[cols].round(3).T.to_string())
    for ds in src.index:
        g = seq.loc[ds]
        best = src.loc[ds, geo].max()
        print(ds, f"E1 adaptm {src.loc[ds, 'adaptm']:.3f} vs adapt {src.loc[ds, 'adapt']:.3f}",
              f"E2 sv_adaptm {src.loc[ds, 'sv_adaptm']:.3f} vs best_geo {best:.3f}",
              f"E3 vs sv_adapt {src.loc[ds, 'sv_adapt']:.3f}",
              f"| seqs adaptm>adapt {(g.adaptm > g.adapt).sum()}/{len(g)}"
              f" sv_adaptm>best_geo {(g.sv_adaptm > g[geo].max(axis=1)).sum()}/{len(g)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "summary"])
    a = ap.parse_args()
    fmap = folds()
    if a.mode == "train":
        train(fmap)
    elif a.mode == "eval":
        evaluate(fmap, prefixes=("adapt", "adaptm"), out_name=OUT)
        summary(pd.read_parquet(RESULTS / f"{OUT}.parquet"))
    else:
        summary(pd.read_parquet(RESULTS / f"{OUT}.parquet"))


if __name__ == "__main__":
    main()
