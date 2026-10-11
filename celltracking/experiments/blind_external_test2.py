"""Second blind test: per-colony processing and label-free selection (SV) on unseen data.

Protocol fixed before download in docs/blind_test2_protocol.md; run once, report as is.
Sources: LYN budding-yeast colonies (lineage_gt.csv + tracked masks in HDF5) and
Kunzmann et al. B. subtilis outgrowth (CTC man_track.txt + masks).

  python experiments/blind_external_test2.py --check   # data sanity only (counts, no predictions)
  python experiments/blind_external_test2.py           # the single blind run
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.blind_external_test import ctc_ancestor, load_ctc, window  # noqa: E402
from experiments.blind_percolony_posthoc import per_colony  # noqa: E402
from experiments.family_set import RESULTS, ROOT  # noqa: E402
from experiments.label_free_selection import score  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402
from experiments.raft_external_eval import bbox_shape, plain  # noqa: E402

LYN = ROOT / "data/raw/lyn_yeast/blind"
BSUB = ROOT / "data/raw/bsub_kunzmann/dataset_b_subtilis"
# colony006 (External) excluded: no frame interval stated in the LYN repository (protocol)
LYN_COLONIES = ["SJR1", "SJR2", "SJR3", "SJR4", "SJR6", "SJR7"]
TAGS = ["raft_sim_loc", "raft_sim_loc_s1", "raft_sim_loc_s2"]
GAP_MINUTES = (30, 60, 120)
SEED = 20261003


def load_lyn(folder):
    seg = next(p for p in sorted(folder.glob("*.h5")) if "segmentation" in p.name)
    csv = next(folder.glob("*lineage_gt.csv"))
    lin = pd.read_csv(csv, comment="#", header=None, names=["parent", "bud", "t", "conf"])
    parent = dict(zip(lin.bud.astype(int), lin.parent.astype(int)))
    frames, keys = {}, {}
    with h5py.File(seg) as f:
        g = f["FOV0"]
        for name in g:
            t = int(name[1:])
            m = g[name][()]
            ids = np.unique(m)
            ids = ids[ids > 0]
            if len(ids):
                frames[t] = np.array(ndimage.center_of_mass(np.ones_like(m), m, ids)).reshape(-1, 2)
                keys[t] = ids.astype(int)
    present = {t: set(k.tolist()) for t, k in keys.items()}

    def ancestor(t, o, a):
        seen = 0
        while o not in present.get(a, ()):
            o = parent.get(o, -1)
            seen += 1
            if o <= 0 or seen > 1000:
                return None
        return o
    return frames, keys, ancestor


def movies():
    for c in LYN_COLONIES:
        frames, keys, anc = load_lyn(LYN / c)
        yield "LYN", c, 5.0, frames, keys, anc
    for tf in sorted(BSUB.glob("*_GT/TRA/man_track.txt")):
        masks = sorted(tf.parent.glob("man_track*.tif"))
        frames, keys, tracks = load_ctc(tf, masks)
        frames = {t: f for t, f in frames.items() if len(f)}
        keys = {t: keys[t] for t in frames}
        yield "BSUB", tf.parent.parent.name[:2], 3.0 if len(masks) == 49 else 4.0, frames, keys, \
            lambda t, o, a, tr=tracks: ctc_ancestor(o, a, tr)


def hypotheses(e, models, minutes, shape):
    out = dict(geometry_assignments(e))
    col, _ = per_colony(e, models, minutes, shape)
    out.update(col)
    fa, fb = plain(e["anchor"]), plain(e["target"])
    na = len(fa["pos"])
    for tag, (model, margs) in models.items():
        if na >= 2:
            prob, _, _ = predict(model, fa, fb, minutes, shape, no_shape=True, init=margs.get("init", "radial"))
            out[tag] = prob[:, :-1].argmax(1)
        else:
            out[tag] = np.zeros(len(fb["pos"]), int)
        cand = [score(out["nearest_anchor"], na), score(out[f"col_{tag}"], na)]
        k = max(range(2), key=lambda i: (cand[i]["coverage"], -cand[i]["size_cv"]))
        out[f"sv_{tag}"] = out["nearest_anchor"] if k == 0 else out[f"col_{tag}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="data sanity only; no predictions")
    a = ap.parse_args()
    models = None if a.check else {t: load(t) for t in TAGS}
    rng = np.random.default_rng(SEED)
    rows = []
    for source, name, mpf, frames, keys, anc in movies():
        ts = sorted(frames)
        shape = bbox_shape(list(frames.values()))
        gaps = {m: int(round(m / mpf)) for m in GAP_MINUTES if round(m / mpf) < len(ts)}
        n_scored = {}
        for minutes, g in gaps.items():
            n_scored[minutes] = 0
            for s in ts:
                if s + g not in frames:
                    continue
                e, valid, truth = window(frames, keys, s, s + g, lambda o, s=s, b=s + g: anc(b, o, s), rng)
                n_scored[minutes] += len(truth)
                if models is None or not len(truth):
                    continue
                for method, p in hypotheses(e, models, minutes, shape).items():
                    rows.append({"dataset": source, "sequence": name, "window_id": f"{s}_{s + g}",
                                 "gap_minutes": minutes, "n_anchors": len(e["anchor"]), "method": method,
                                 "accuracy": float(np.mean(p[valid] == truth))})
        sizes = [len(f) for f in frames.values()]
        print(source, name, f"mpf={mpf} frames={len(ts)} cells {min(sizes)}-{max(sizes)} gaps={gaps} "
              f"scored={n_scored}", flush=True)
    if models is None:
        return
    d = pd.DataFrame(rows)
    d.to_parquet(RESULTS / "blind_external_test2.parquet", index=False)
    seq = d.groupby(["dataset", "sequence", "gap_minutes", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    for p in ("", "col_", "sv_"):
        seq[f"{p}sim3"] = seq[[f"{p}{t}" for t in TAGS]].mean(axis=1)
    geo = [c for c in seq if any(g in c for g in ("nearest_anchor", "centroid_translation",
                                                  "radial_expansion", "radial_sector"))]
    seq["best_geo"] = seq[geo].max(axis=1)
    cols = geo + ["sim3", "col_sim3", "sv_sim3", "best_geo"]
    print(seq[cols].round(3).to_string())
    src = seq.groupby("dataset").mean()
    print(src[cols].round(3).T.to_string())
    for ds, g in seq.groupby("dataset"):
        best = src.loc[ds, geo].max()
        print(ds, f"H1 SV {src.loc[ds, 'sv_sim3']:.3f} vs best geo {best:.3f} ({src.loc[ds, geo].idxmax()})",
              f"H2 {(g.sv_sim3 > g.best_geo).sum()}/{len(g)}",
              f"H3 SV-nearest {src.loc[ds, 'sv_sim3'] - src.loc[ds, 'nearest_anchor']:+.3f} "
              f"SV-col {src.loc[ds, 'sv_sim3'] - src.loc[ds, 'col_sim3']:+.3f}")


if __name__ == "__main__":
    main()
