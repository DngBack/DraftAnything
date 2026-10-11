"""Blind test 4 (docs/blind_test4_protocol.md): unused van Vliet subsets trpL, metA, pheA.

  python experiments/blind_test4.py train
  python experiments/blind_test4.py eval
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS, ROOT  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.ot_baseline import couple  # noqa: E402
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402
from experiments.raft_external_eval import bbox_shape, load_schnitz, plain, reca_window  # noqa: E402
from experiments.self_label_adapt import BlindPseudoMovie, train_movies  # noqa: E402
from experiments.self_label_adapt_ext import model_links  # noqa: E402

SUBSETS = ("trpL", "metA", "pheA")
GAPS = (4, 8, 16)
MPF = 7.5
OT = {"ot_exact": ("exact", {"c": 1.0}), "ot_uot": ("uot", {"eps": 0.5, "rho": 10.0})}
BASELINES = ("nearest_anchor", "centroid_translation", "radial_expansion", "radial_sector", *OT)


def colonies(subset):
    base = ROOT / "data/raw" / f"vv_{subset}"
    out = {}
    for d in sorted(base.iterdir()):
        try:
            out[d.name] = load_schnitz(d, skip_bad_rect=True)
        except Exception as err:  # excluded and reported, as fixed in the protocol
            print("EXCLUDED", subset, d.name, repr(err), flush=True)
    return out


def fold_of(names):
    return {n: i % 2 for i, n in enumerate(sorted(names))}


def train():
    sim = load("raft_sim_loc")[0]
    for subset in SUBSETS:
        cols = colonies(subset)
        fmap = fold_of(cols)
        yx = {c: {t: f[:, ::-1] for t, f in v[0].items()} for c, v in cols.items()}
        links = {c: model_links(yx[c], MPF, sim) for c in cols}
        for fold in (0, 1):
            other = [c for c in cols if fmap[c] != fold]
            train_movies(f"bt4_adapt_{subset}_{fold}", [BlindPseudoMovie(yx[c], MPF) for c in other])
            train_movies(f"bt4_adaptm_{subset}_{fold}", [BlindPseudoMovie(yx[c], MPF, links[c]) for c in other])


def evaluate():
    sim = load("raft_sim_loc")
    rng = np.random.default_rng(20261002)
    rows = []
    for subset in SUBSETS:
        cols = colonies(subset)
        fmap = fold_of(cols)
        adapt = {f: load(f"bt4_adapt_{subset}_{f}") for f in (0, 1)}
        adaptm = {f: load(f"bt4_adaptm_{subset}_{f}") for f in (0, 1)}
        for colony, (frames, mapping, tracks, invalid, _) in cols.items():
            shape = bbox_shape([f[:, ::-1] for f in frames.values() if len(f)])
            models = {"sim": sim, "adapt": adapt[fmap[colony]], "adaptm": adaptm[fmap[colony]]}
            for gap in GAPS:
                for start in sorted(frames):
                    if start + gap not in frames or not len(frames[start]) or not len(frames[start + gap]):
                        continue
                    e, valid, truth = reca_window(start, gap, frames, mapping, tracks, invalid, rng)
                    if not len(truth):
                        continue
                    out = dict(geometry_assignments(e))
                    src = e["anchor"][["center_y", "center_x"]].to_numpy(float)
                    dst = e["target"][["center_y", "center_x"]].to_numpy(float)
                    for name, (kind, p) in OT.items():
                        out[name] = couple(src, dst, kind, p) if len(src) >= 2 else np.zeros(len(dst), int)
                    fa, fb = plain(e["anchor"]), plain(e["target"])
                    for tag, (model, _) in models.items():
                        out[tag] = (predict(model, fa, fb, gap * MPF, shape, no_shape=True)[0][:, :-1].argmax(1)
                                    if len(fa["pos"]) >= 2 else np.zeros(len(fb["pos"]), int))
                    for method, p in out.items():
                        rows.append({"dataset": subset, "sequence": colony, "gap": gap,
                                     "window_id": f"{start}_{start + gap}", "n_anchors": len(src),
                                     "method": method, "accuracy": float(np.mean(p[valid] == truth))})
            print(subset, colony, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(RESULTS / "blind_test4.parquet", index=False)
    summary(d)


def summary(d):
    seq = d.groupby(["dataset", "sequence", "gap", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    src = seq.groupby("dataset").mean()
    print(src[list(BASELINES) + ["sim", "adapt", "adaptm"]].round(3).to_string())
    for ds in SUBSETS:
        g = seq.loc[ds]
        best = src.loc[ds, list(BASELINES)]
        print(ds, f"F1 sim {src.loc[ds, 'sim']:.3f} vs best {best.max():.3f} ({best.idxmax()})",
              f"F2 colonies {(g.sim > g[list(BASELINES)].max(axis=1)).sum()}/{len(g)}",
              f"F3 adaptm {src.loc[ds, 'adaptm']:.3f} vs sim", f"F4 vs adapt {src.loc[ds, 'adapt']:.3f}",
              f"| colonies adaptm>sim {(g.adaptm > g.sim).sum()}/{len(g)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "summary"])
    a = ap.parse_args()
    {"train": train, "eval": evaluate,
     "summary": lambda: summary(pd.read_parquet(RESULTS / "blind_test4.parquet"))}[a.mode]()


if __name__ == "__main__":
    main()
