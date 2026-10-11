"""Protocol 3 (docs/blind_test3_protocol.md): annotation-free self-labelled adaptation.

For each external source, the sim-trained model is fine-tuned on chained one-frame
pseudo-lineage of the other fold's unannotated sequences, then scored on the exact
windows of blind tests 1 and 2.

  python experiments/self_label_adapt.py train
  python experiments/self_label_adapt.py eval
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import blind_external_test as b1, blind_external_test2 as b2  # noqa: E402
from experiments import lineage_raft  # noqa: E402
from experiments.blind_percolony_posthoc import per_colony  # noqa: E402
from experiments.chained_pseudo_lineage import chain_positions, compose  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402
from experiments.label_free_selection import score  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402
from experiments.raft_external_eval import bbox_shape  # noqa: E402

SOURCES = ["TrackRefiner", "TrackMate", "LYN", "BSUB"]


_CACHE = []


def all_movies():
    """(source, name, minutes per frame, frames, keys, ancestor) for both blind tests, loaded once."""
    if not _CACHE:
        for gen in (b1.movies(), b2.movies()):
            _CACHE.extend(gen)
    return _CACHE


def folds():
    names = {}
    for source, name, *_ in all_movies():
        names.setdefault(source, []).append(name)
    return {(s, n): i % 2 for s, ns in names.items() for i, n in enumerate(sorted(ns))}


class BlindPseudoMovie:
    """Duck-types lineage_glue.Movie; frames keyed by minute, lineage from chained links."""

    def __init__(self, frames, mpf, links=None):
        self.links = links if links is not None else chain_positions(frames)
        self.key2t = {int(round(t * mpf)): t for t in sorted(frames)}
        self.frames = {k: {"pos": frames[t], "axis": np.zeros((len(frames[t]), 2)),
                           "major": np.ones(len(frames[t])), "minor": np.ones(len(frames[t]))}
                       for k, t in self.key2t.items()}
        self.image_shape = bbox_shape(list(frames.values()))
        self.valid_a = [k for k, f in self.frames.items() if len(f["pos"]) >= 2]

    def window(self, a, b):
        return self.frames[a], self.frames[b], compose(self.links, self.key2t[a], self.key2t[b],
                                                      len(self.frames[b]["pos"]))


def train(source, fold, fmap):
    tag = f"adapt_{source}_{fold}"
    if (lineage_raft.CKPT / f"{tag}.done").exists():
        return
    movies = [BlindPseudoMovie(frames, mpf) for s, name, mpf, frames, _, _ in all_movies()
              if s == source and fmap[(s, name)] != fold]
    train_movies(tag, movies)


def train_movies(tag, movies):
    """Fine-tune raft_sim_loc on pseudo-lineage movies (protocol 3 settings)."""
    done = lineage_raft.CKPT / f"{tag}.done"
    if done.exists():
        return
    print(tag, "training sequences", len(movies), flush=True)
    args = Namespace(mode="train", train=[], sim="", n_sim=0, tag=tag, steps=6000, lr=2e-4, d=128, iters=4,
                     closs=0.3, family=0.5, consist=0.0, hmin=15, hmax=240, no_shape=True, rot="full",
                     init="radial", workers=4, tta=1, seed=0, sim_full_fov=False, init_from="raft_sim_loc",
                     pseudo=False, eval=[], horizons=[])
    lineage_raft.load_movies = lambda _a: movies
    lineage_raft.train(args)
    done.touch()


def evaluate(fmap, prefixes=("adapt",), out_name="self_label_adapt"):
    sim = load("raft_sim_loc")
    adapted = {(p, s, f): load(f"{p}_{s}_{f}") for p in prefixes for s in SOURCES for f in (0, 1)}
    rows = []
    for test, seed in (("blind1", b1.SEED), ("blind2", b2.SEED)):
        rng = np.random.default_rng(seed)
        for source, name, mpf, frames, keys, anc in all_movies():
            if (source in ("TrackRefiner", "TrackMate")) != (test == "blind1"):
                continue
            ts = sorted(frames)
            shape = bbox_shape(list(frames.values()))
            gaps = {m: int(round(m / mpf)) for m in b1.GAP_MINUTES if round(m / mpf) < len(ts)}
            models = {"sim": sim, **{p: adapted[(p, source, fmap[(source, name)])] for p in prefixes}}
            for minutes, g in gaps.items():
                for s in ts:
                    if s + g not in frames:
                        continue
                    e, valid, truth = b1.window(frames, keys, s, s + g, lambda o, s=s, b=s + g: anc(b, o, s), rng)
                    if not len(truth):
                        continue
                    out = dict(geometry_assignments(e))
                    out.update(per_colony(e, models, minutes, shape)[0])
                    fa, fb = b2.plain(e["anchor"]), b2.plain(e["target"])
                    na = len(fa["pos"])
                    for tag, (model, margs) in models.items():
                        out[tag] = (predict(model, fa, fb, minutes, shape, no_shape=True)[0][:, :-1].argmax(1)
                                    if na >= 2 else np.zeros(len(fb["pos"]), int))
                        cand = [score(out["nearest_anchor"], na), score(out[f"col_{tag}"], na)]
                        k = max(range(2), key=lambda i: (cand[i]["coverage"], -cand[i]["size_cv"]))
                        out[f"sv_{tag}"] = out["nearest_anchor"] if k == 0 else out[f"col_{tag}"]
                    for method, p in out.items():
                        rows.append({"test": test, "dataset": source, "sequence": name, "window_id": f"{s}_{s + g}",
                                     "gap_minutes": minutes, "n_anchors": na, "method": method,
                                     "accuracy": float(np.mean(p[valid] == truth))})
            print(source, name, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(RESULTS / f"{out_name}.parquet", index=False)
    summary(d)


def summary(d):
    seq = d.groupby(["dataset", "sequence", "gap_minutes", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    geo = [c for c in seq if any(g in c for g in ("nearest_anchor", "centroid_translation",
                                                  "radial_expansion", "radial_sector"))]
    src = seq.groupby("dataset").mean()
    cols = ["nearest_anchor", "sim", "col_sim", "sv_sim", "adapt", "col_adapt", "sv_adapt"]
    print(src[cols].round(3).T.to_string())
    for ds in src.index:
        best = src.loc[ds, geo].max()
        g = seq.loc[ds]
        print(ds, f"A1 sv_adapt {src.loc[ds, 'sv_adapt']:.3f} vs best_geo {best:.3f} ({src.loc[ds, geo].idxmax()})",
              f"A2 vs sv_sim {src.loc[ds, 'sv_sim']:.3f}", f"A3 adapt {src.loc[ds, 'adapt']:.3f} vs sim "
              f"{src.loc[ds, 'sim']:.3f}", f"| seqs sv_adapt>best_geo {(g.sv_adapt > g[geo].max(axis=1)).sum()}/{len(g)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "summary"])
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    a = ap.parse_args()
    fmap = folds()
    if a.mode == "train":
        for s in a.sources:
            for f in (0, 1):
                train(s, f, fmap)
    elif a.mode == "eval":
        evaluate(fmap)
    else:
        summary(pd.read_parquet(RESULTS / "self_label_adapt.parquet"))


if __name__ == "__main__":
    main()
