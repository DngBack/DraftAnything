"""Protocol 4 (docs/selflabel_ext_protocol.md): self-labelled adaptation on FAST, recA, rpsM.

  python experiments/self_label_adapt_ext.py train
  python experiments/self_label_adapt_ext.py eval
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS, ROOT  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402
from experiments.raft_external_eval import (FAST_GAPS, FAST_HELD_OUT, RECA_GAPS, RECA_HELD_OUT,  # noqa: E402
                                            RPSM_HELD_OUT, bbox_shape, fast_window, load_fast,
                                            load_schnitz, plain, reca_window)
from experiments.self_label_adapt import BlindPseudoMovie, train_movies  # noqa: E402

SETS = (("recA", RECA_HELD_OUT, "reca_holdout", RECA_GAPS, 7.5),
        ("rpsM", RPSM_HELD_OUT, "rpsm_holdout", (10, 20, 40), 3.0),
        ("FAST", FAST_HELD_OUT, None, FAST_GAPS, 5.0))


def load_colony(name, colony, base):
    if name == "FAST":
        frames, mapping, tracks = load_fast(ROOT / "data/downloads" / f"fast_{colony}.zip")[:3]
        return frames, (mapping, tracks)
    frames, mapping, tracks, invalid, _ = load_schnitz(ROOT / "data/raw" / base / colony)
    return frames, (mapping, tracks, invalid)


def fold_of(held):
    return {c: i % 2 for i, c in enumerate(sorted(held))}


def model_links(yx, mpf, model):
    """Protocol 5: one-frame links from the sim model itself (argmax), for composition."""
    shape = bbox_shape([f for f in yx.values() if len(f)])
    ks = sorted(yx)
    links = {ks[0]: -np.ones(len(yx[ks[0]]), int)}
    for a, b in zip(ks[:-1], ks[1:]):
        if len(yx[a]) >= 2 and len(yx[b]):
            fa = {"pos": yx[a], "axis": np.zeros((len(yx[a]), 2)), "major": np.ones(len(yx[a])), "minor": np.ones(len(yx[a]))}
            fb = {"pos": yx[b], "axis": np.zeros((len(yx[b]), 2)), "major": np.ones(len(yx[b])), "minor": np.ones(len(yx[b]))}
            links[b] = predict(model, fa, fb, (b - a) * mpf, shape, no_shape=True)[0][:, :-1].argmax(1)
        else:
            links[b] = np.zeros(len(yx[b]), int) if len(yx[a]) == 1 else -np.ones(len(yx[b]), int)
    return links


def train(prefix="adapt"):
    sim = load("raft_sim_loc")[0] if prefix == "adaptm" else None
    for name, held, base, _, mpf in SETS:
        fmap = fold_of(held)
        data = {c: {t: f[:, ::-1] for t, f in load_colony(name, c, base)[0].items()} for c in held}
        links = {c: model_links(data[c], mpf, sim) for c in held} if sim is not None else {c: None for c in held}
        for fold in (0, 1):
            movies = [BlindPseudoMovie(data[c], mpf, links[c]) for c in held if fmap[c] != fold]
            train_movies(f"{prefix}_{name}_{fold}", movies)


def evaluate():
    sim = load("raft_sim_loc")
    rng = np.random.default_rng(20261002)
    rows = []
    for name, held, base, gaps, mpf in SETS:
        fmap = fold_of(held)
        adapt = {f: load(f"adapt_{name}_{f}") for f in (0, 1)}
        adaptm = {f: load(f"adaptm_{name}_{f}") for f in (0, 1)}
        for colony in held:
            frames, aux = load_colony(name, colony, base)
            shape = bbox_shape([f[:, ::-1] for f in frames.values() if len(f)])
            models = {"sim": sim, "adapt": adapt[fmap[colony]], "adaptm": adaptm[fmap[colony]]}
            for gap in gaps:
                for start in sorted(frames):
                    if start + gap not in frames:
                        continue
                    if name == "FAST":
                        e, valid, truth = fast_window(start, gap, frames, *aux, rng)
                    else:
                        if not len(frames[start]) or not len(frames[start + gap]):
                            continue
                        e, valid, truth = reca_window(start, gap, frames, *aux, rng)
                    if not len(truth):
                        continue
                    out = dict(geometry_assignments(e))
                    fa, fb = plain(e["anchor"]), plain(e["target"])
                    for tag, (model, _) in models.items():
                        out[tag] = (predict(model, fa, fb, gap * mpf, shape, no_shape=True)[0][:, :-1].argmax(1)
                                    if len(fa["pos"]) >= 2 else np.zeros(len(fb["pos"]), int))
                    for method, p in out.items():
                        rows.append({"dataset": name, "sequence": colony, "gap": gap, "window_id": f"{start}_{start+gap}",
                                     "method": method, "accuracy": float(np.mean(p[valid] == truth))})
            print(name, colony, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(RESULTS / "self_label_adapt_ext2.parquet", index=False)
    summary(d)


def summary(d):
    seq = d.groupby(["dataset", "sequence", "gap", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    geo = ["nearest_anchor", "centroid_translation", "radial_expansion", "radial_sector"]
    src = seq.groupby("dataset").mean()
    m = [c for c in ("sim", "adapt", "adaptm") if c in src]
    print(src[geo + m].round(3).to_string())
    for ds in src.index:
        g = seq.loc[ds]
        print(ds, f"B1 adapt {src.loc[ds, 'adapt']:.3f} vs sim {src.loc[ds, 'sim']:.3f}",
              f"B2 vs best_geo {src.loc[ds, geo].max():.3f} | seqs adapt>sim {(g.adapt > g.sim).sum()}/{len(g)}")
        if "adaptm" in g:
            print(ds, f"C1 adaptm {src.loc[ds, 'adaptm']:.3f} vs sim {src.loc[ds, 'sim']:.3f}",
                  f"C2 vs best_geo {src.loc[ds, geo].max():.3f} | seqs adaptm>sim {(g.adaptm > g.sim).sum()}/{len(g)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "summary"])
    ap.add_argument("--prefix", default="adapt", choices=["adapt", "adaptm"])
    a = ap.parse_args()
    if a.mode == "train":
        train(a.prefix)
    elif a.mode == "eval":
        evaluate()
    else:
        summary(pd.read_parquet(RESULTS / "self_label_adapt_ext.parquet"))


if __name__ == "__main__":
    main()
