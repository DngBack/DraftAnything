"""Blind external test on lineage data never used in development.

Protocol fixed beforehand in docs/blind_test_protocol.md; run once, report as is.
Sources: TrackRefiner ground truth (CellProfiler CSV with parent links) and the
TrackMate 8 E. coli protocol data (CTC man_track.txt + label masks).

  python experiments/blind_external_test.py --check   # data sanity only (counts, no predictions)
  python experiments/blind_external_test.py           # the single blind run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS, ROOT  # noqa: E402
from experiments.raft_external_eval import Runner, bbox_shape, summary  # noqa: E402

TR = ROOT / "data/raw/trackrefiner/repo/09.GroundTruth"
TM = ROOT / "data/raw/trackmate_ecoli/Protocol to study the growth and morphology of Escherichia coli with TrackMate"
# (name, file, minutes per frame); Constrained_1 / unconstrained_4 files are swapped in the repo
TR_MOVIES = [("TR-Unc1", "ground_truth_unconstrained_1/unconstrained_1.GT.csv", 2.0),
             ("TR-Unc2", "ground_truth_unconstrained_2/unconstrained_2.GT.csv", 2.0),
             ("TR-Unc3", "ground_truth_unconstrained_3/unconstrained_3.GT.csv", 2.0),
             ("TR-Unc4", "ground_truth_constrained_1/Constrained_1.GT.csv", 3.0),
             ("TR-Con1", "ground_truth_unconstrained_4/unconstrained_4.GT.csv", 10.0),
             ("TR-Con2", "ground_truth_constrained_2/Constrained_2.GT.csv", 5.0),
             ("TR-Con3", "ground_truth_constrained_3/Constrained_3.GT.csv", 1.75),
             ("TR-Con4", "ground_truth_constrained_4/Constrained_4.GT.csv", 1.75)]
# curated CTC exports; UbiH/SecondaryOptimization (same masks, re-run links) is not used
TM_MOVIES = [("TM-WT", "3bis-Optimization results"), ("TM-UbiG", "3ter-CrossValidation/UbiG"),
             ("TM-UbiH", "3ter-CrossValidation/UbiH")]
GAP_MINUTES = (30, 60, 120)
SEED = 20261002


def load_trackrefiner(path):
    """frames[t] = (N,2) array of (y, x); key[t] = object numbers; parent[(t, obj)] = (t', obj') or None."""
    d = pd.read_csv(path, usecols=["ImageNumber", "ObjectNumber", "AreaShape_Center_X", "AreaShape_Center_Y",
                                   "TrackObjects_ParentImageNumber_50", "TrackObjects_ParentObjectNumber_50"])
    frames, keys, parent = {}, {}, {}
    for t, g in d.groupby("ImageNumber"):
        frames[int(t)] = g[["AreaShape_Center_Y", "AreaShape_Center_X"]].to_numpy(float)
        keys[int(t)] = g.ObjectNumber.to_numpy(int)
    for t, o, pt, po in zip(d.ImageNumber, d.ObjectNumber, d.TrackObjects_ParentImageNumber_50,
                            d.TrackObjects_ParentObjectNumber_50):
        parent[(int(t), int(o))] = (int(pt), int(po)) if pt > 0 else None
    return frames, keys, parent


def tr_ancestor(t, o, a, parent):
    """Follow parent links from (t, o) back to frame a; None if the chain breaks or skips frame a."""
    while t > a:
        p = parent.get((t, o))
        if p is None or p[0] < a:
            return None
        t, o = p
    return o if t == a else None


def load_ctc(track_file, mask_files):
    """frames[t] = (N,2) centroids; keys[t] = labels; tracks[label] = (begin, end, parent)."""
    tracks = {}
    for line in open(track_file):
        if line.strip():
            lab, b, e, p = map(int, line.split())
            tracks[lab] = (b, e, p)
    frames, keys = {}, {}
    for t, f in enumerate(mask_files):
        m = tifffile.imread(f)
        ids = np.unique(m)
        ids = ids[ids > 0]
        frames[t] = np.array(ndimage.center_of_mass(np.ones_like(m), m, ids)).reshape(-1, 2)
        keys[t] = ids.astype(int)
    return frames, keys, tracks


def ctc_ancestor(lab, a, tracks):
    while lab in tracks and tracks[lab][0] > a:
        lab = tracks[lab][2]
        if lab == 0:
            return None
    if lab not in tracks:
        return None
    b, e, _ = tracks[lab]
    return lab if b <= a <= e else None


def window(frames, keys, a, b, ancestor, rng):
    sa, sb = rng.permutation(len(frames[a])), rng.permutation(len(frames[b]))
    pa, pb = frames[a][sa], frames[b][sb]
    ka, kb = keys[a][sa], keys[b][sb]
    index = {int(k): i for i, k in enumerate(ka)}
    valid, truth = [], []
    for j, k in enumerate(kb):
        anc = ancestor(int(k))
        if anc is not None and anc in index:
            valid.append(j)
            truth.append(index[anc])
    e = {"anchor": pd.DataFrame({"local_detection_id": np.arange(len(pa)), "center_y": pa[:, 0], "center_x": pa[:, 1]}),
         "target": pd.DataFrame({"local_detection_id": np.arange(len(pb)), "center_y": pb[:, 0], "center_x": pb[:, 1]}),
         "cost": np.sum((pb[:, None] - pa[None]) ** 2, -1).astype(np.float32)}
    return e, np.array(valid, int), np.array(truth, int)


def movies():
    for name, rel, mpf in TR_MOVIES:
        frames, keys, parent = load_trackrefiner(TR / rel)
        yield "TrackRefiner", name, mpf, frames, keys, lambda t, o, a, p=parent: tr_ancestor(t, o, a, p)
    for name, rel in TM_MOVIES:
        tf = TM / rel / "01_ST/TRA/man_track.txt"
        masks = sorted(tf.parent.glob("man_track*.tif"))
        frames, keys, tracks = load_ctc(tf, masks)
        yield "TrackMate", name, 2.0, frames, keys, lambda t, o, a, tr=tracks: ctc_ancestor(o, a, tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="data sanity only; no predictions")
    ap.add_argument("--tags", nargs="+", default=["raft_sim_loc", "raft_sim_loc_s1", "raft_sim_loc_s2", "raft_v1_loc"])
    a = ap.parse_args()
    runner = None if a.check else Runner(a.tags, ["argmax"])
    rng = np.random.default_rng(SEED)
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
                e["window"] = {"horizon_frames": minutes}
                n_scored[minutes] += len(truth)
                if runner is not None and len(truth):
                    runner.window(source, name, f"{s}_{s + g}", minutes, e, valid, truth, shape)
        sizes = [len(f) for f in frames.values()]
        print(source, name, f"frames={len(ts)} cells {min(sizes)}-{max(sizes)} gaps(frames)={gaps} "
              f"scored targets={n_scored}", flush=True)
    if runner is not None:
        out = pd.DataFrame(runner.rows)
        out.to_parquet(RESULTS / "blind_external_test.parquet", index=False)
        summary(out)


if __name__ == "__main__":
    main()
