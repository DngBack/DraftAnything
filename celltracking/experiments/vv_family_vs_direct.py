"""Family registration vs direct matcher in domain on van Vliet (docs/vv_family_vs_direct_protocol.md).

  python experiments/vv_family_vs_direct.py train
  python experiments/vv_family_vs_direct.py eval
  python experiments/vv_family_vs_direct.py summary
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import lineage_raft  # noqa: E402
from experiments.blind_test4 import GAPS, MPF, SUBSETS, colonies, fold_of  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.raft_external_eval import bbox_shape, plain, reca_window  # noqa: E402
from experiments.self_label_adapt import BlindPseudoMovie  # noqa: E402
from experiments.self_label_adapt_ext import model_links  # noqa: E402

OUT = RESULTS / "vv_family_vs_direct.parquet"


def train_dm(tag, movies):
    """bt4_adaptm settings, but the direct matcher initialised from dm_sim_loc_lr."""
    done = lineage_raft.CKPT / f"{tag}.done"
    if done.exists():
        return
    print(tag, "training sequences", len(movies), flush=True)
    args = Namespace(mode="train", train=[], sim="", n_sim=0, tag=tag, steps=6000, lr=2e-4, d=128, iters=4,
                     closs=0.3, family=0.5, consist=0.0, hmin=15, hmax=240, no_shape=True, rot="full",
                     init="radial", workers=4, tta=1, seed=0, sim_full_fov=False, init_from="dm_sim_loc_lr",
                     pseudo=False, eval=[], horizons=[], no_centre=True, aux_centre=False, k_out=24)
    lineage_raft.load_movies = lambda _a: movies
    lineage_raft.train(args)
    done.touch()


def train():
    sim = load("raft_sim_loc")[0]  # pseudo-lineage identical to bt4_adaptm
    for subset in SUBSETS:
        cols = colonies(subset)
        fmap = fold_of(cols)
        yx = {c: {t: f[:, ::-1] for t, f in v[0].items()} for c, v in cols.items()}
        links = {c: model_links(yx[c], MPF, sim) for c in cols}
        for fold in (0, 1):
            other = [c for c in cols if fmap[c] != fold]
            train_dm(f"bt4dm_adaptm_{subset}_{fold}", [BlindPseudoMovie(yx[c], MPF, links[c]) for c in other])


def evaluate():
    rng = np.random.default_rng(20261002)  # same windows as blind_test4.evaluate
    sim, dm_sim = load("raft_sim_loc"), load("dm_sim_loc_lr")
    rows = []
    for subset in SUBSETS:
        cols = colonies(subset)
        fmap = fold_of(cols)
        adaptm = {f: load(f"bt4_adaptm_{subset}_{f}") for f in (0, 1)}
        dm_adaptm = {f: load(f"bt4dm_adaptm_{subset}_{f}") for f in (0, 1)}
        for colony, (frames, mapping, tracks, invalid, _) in cols.items():
            shape = bbox_shape([f[:, ::-1] for f in frames.values() if len(f)])
            f = fmap[colony]
            models = {"sim": sim, "adaptm": adaptm[f], "dm_sim": dm_sim, "dm_adaptm": dm_adaptm[f]}
            for gap in GAPS:
                for start in sorted(frames):
                    if start + gap not in frames or not len(frames[start]) or not len(frames[start + gap]):
                        continue
                    e, valid, truth = reca_window(start, gap, frames, mapping, tracks, invalid, rng)
                    if not len(truth):
                        continue
                    fa, fb = plain(e["anchor"]), plain(e["target"])
                    for tag, (model, _) in models.items():
                        p = (predict(model, fa, fb, gap * MPF, shape, no_shape=True)[0][:, :-1].argmax(1)
                             if len(fa["pos"]) >= 2 else np.zeros(len(fb["pos"]), int))
                        rows.append({"dataset": subset, "sequence": colony, "gap": gap,
                                     "window_id": f"{start}_{start + gap}", "n_anchors": len(fa["pos"]),
                                     "method": tag, "accuracy": float(np.mean(p[valid] == truth))})
            print(subset, colony, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT, index=False)
    summary(d)


def colony_table(d):
    return d.groupby(["dataset", "sequence", "gap", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")


def boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def summary(d):
    methods = ["sim", "adaptm", "dm_sim", "dm_adaptm"]
    seq = colony_table(d)[methods]
    print("colonies", len(seq))
    print(seq.groupby("dataset").mean().round(3).to_string())
    print("all", seq.mean().round(3).to_dict())
    diff = (seq.adaptm - seq.dm_adaptm).to_numpy()
    m, lo, hi = boot(diff)
    print(f"V1 adaptm - dm_adaptm = {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'HOLDS' if lo > 0 else 'FAILS'};"
          f" colonies won {(diff > 0).sum()}/{len(diff)}")
    sub = seq.groupby("dataset").mean()
    v2 = (sub.adaptm > sub.dm_adaptm)
    print("V2 per subset adaptm > dm_adaptm:", v2.to_dict(), "->", "HOLDS" if v2.all() else "FAILS")
    m, lo, hi = boot((seq.sim - seq.dm_sim).to_numpy())
    print(f"zero-shot sim - dm_sim = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("self-labelling gain: LineageRAFT", round((seq.adaptm - seq.sim).mean(), 3),
          "direct matcher", round((seq.dm_adaptm - seq.dm_sim).mean(), 3))
    per_gap = d.groupby(["dataset", "sequence", "gap", "method"]).accuracy.mean().unstack("method")[methods]
    print("per gap (mean over colonies)\n", per_gap.groupby("gap").mean().round(3).to_string())
    big = d[d.n_anchors > 100]
    if len(big):
        print("windows > 100 ancestors:", big.window_id.nunique(), "\n",
              big.groupby(["gap", "method"]).accuracy.mean().unstack()[methods].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "summary"])
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "summary": lambda: summary(pd.read_parquet(OUT))}[a.mode]()


if __name__ == "__main__":
    main()
