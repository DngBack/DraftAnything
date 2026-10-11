"""Locked OIAM -> seven FAST microcolony transfer test.

Protocol: docs/261002_fast_microcolony_external_protocol.md. The first FAST
colony may be used with --audit-only to check MATLAB parsing. The other seven
colonies must not be read until this code is fixed. Tracks, cell IDs and
lineage are used only in the evaluator; inference uses endpoint centroids.

Run from celltracking/:
  python experiments/fast_microcolony_external_test.py --audit-only
  python experiments/fast_microcolony_external_test.py
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402
from experiments.pairwise_geometry_baseline import (HORIZONS, pair_assignment,
                                                    train_classifier)  # noqa: E402

DEV = "140408_01_cib"
HELD_OUT = ("140408_02_cib", "140408_09_cib", "140408_10_cib",
            "140408_11_cib", "140409_03_cib", "140415_08_cib",
            "140415_13_cib")
GAPS = (7, 13, 26)
MINUTES_PER_FRAME = 5  # Approximation fixed in protocol; no archive timestamp.


def _mat_bytes(archive, basename):
    matches = [x for x in archive.namelist() if x.endswith("/" + basename)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {basename}, found {len(matches)}")
    return archive.read(matches[0])


def _matlab_rows(value):
    a = np.asarray(value)
    if a.size == 0:
        return np.empty((0, 2))
    return a.reshape(-1, 2)


def _load_gold(blob):
    try:
        return scipy.io.loadmat(io.BytesIO(blob), simplify_cells=True)
    except NotImplementedError:
        # Some FAST gold-standard files are MATLAB v7.3 HDF5, others v5.
        with h5py.File(io.BytesIO(blob), "r") as f:
            group = f["procTracks"]
            n_tracks = group["times"].size
            tracks = []
            for i in range(n_tracks):
                tr = {}
                for field in ("times", "x", "y", "M"):
                    ds = group[field]
                    raw = np.asarray(f[ds[i, 0]][()]).ravel()
                    if field == "M" and raw.shape == (2,) and np.all(raw == 0):
                        raw = np.array([])
                    tr[field] = raw
                tracks.append(tr)
            mappings = []
            ds = f["toMappings"]
            for t in range(ds.shape[1]):
                raw = np.asarray(f[ds[0, t]][()])
                if raw.size == 0:
                    mappings.append(np.empty((0, 2)))
                elif raw.ndim == 2 and raw.shape[0] == 2:
                    mappings.append(raw.T)
                else:
                    raise ValueError("Unexpected HDF5 toMappings shape")
            return {"procTracks": tracks, "toMappings": mappings}


def load_colony(path):
    with zipfile.ZipFile(path) as archive:
        gold = _load_gold(_mat_bytes(archive, "Tracks_GoldStandardDivisions.mat"))
        tracks = gold["procTracks"]
        if not isinstance(tracks, (list, np.ndarray)):
            tracks = [tracks]
        if len(gold["toMappings"]) == 0:
            raise ValueError("Missing per-frame mappings")
        def observations(track_records):
            return {(int(t), round(float(x), 7), round(float(y), 7))
                    for tr in track_records
                    for t, x, y in zip(np.atleast_1d(tr["times"]),
                                       np.atleast_1d(tr["x"]),
                                       np.atleast_1d(tr["y"]))}
        gold_observations = observations(tracks)
        with h5py.File(io.BytesIO(_mat_bytes(archive,
                 "Tracks_ManualSeg_Corrected.mat")), "r") as corrected:
            manual = []
            n_manual = corrected["procTracks/times"].size
            for i in range(n_manual):
                tr = {}
                for feature in ("times", "x", "y"):
                    refs = corrected[f"procTracks/{feature}"]
                    tr[feature] = np.asarray(corrected[refs[i, 0]][()]).ravel()
                manual.append(tr)
            manual_observations = observations(manual)
            if not manual_observations.issubset(gold_observations):
                raise ValueError("Gold-standard tracks lost manually corrected observations")
            extra_gold_observations = len(gold_observations - manual_observations)
        with h5py.File(io.BytesIO(_mat_bytes(archive,
                 "CellFeatures_ManualSeg.mat")), "r") as features:
            centers = features["trackableData/Centroid"]
            if centers.shape[0] != 1 or centers.shape[1] != len(gold["toMappings"]):
                raise ValueError("Centroid/mapping frame count mismatch")
            frames = {}
            mappings = {}
            validated = 0
            unmapped = 0
            age_index_mismatches = 0
            for frame in range(1, centers.shape[1] + 1):
                c = np.asarray(features[centers[0, frame - 1]][()]).T
                m = _matlab_rows(gold["toMappings"][frame - 1])
                if len(c) != len(m):
                    raise ValueError(f"Frame {frame}: centroid/mapping count mismatch")
                if c.shape[1] != 2:
                    raise ValueError("Centroid must be 2-D")
                if not np.all(np.isfinite(c)):
                    raise ValueError("Nonfinite centroid")
                mapped_ids = [int(row[0]) for row in m if np.isfinite(row[0])]
                if len(mapped_ids) != len(set(mapped_ids)):
                    raise ValueError("Same corrected track appears twice in one frame")
                frames[frame] = c
                mappings[frame] = m
                for row, (track_id, age_index) in enumerate(m):
                    if not np.isfinite(track_id):
                        unmapped += 1
                        continue
                    tid = int(track_id)
                    age = int(age_index)
                    if tid < 1 or tid > len(tracks) or age < 1:
                        raise ValueError("Invalid track mapping")
                    tr = tracks[tid - 1]
                    times = np.atleast_1d(tr["times"])
                    hits = np.where(times == frame)[0]
                    if len(hits) != 1:
                        raise ValueError(f"Track {tid} does not occur exactly once at frame {frame}")
                    if age != int(hits[0]) + 1:
                        age_index_mismatches += 1
                    expected = [np.atleast_1d(tr["x"])[hits[0]],
                                np.atleast_1d(tr["y"])[hits[0]]]
                    if not np.allclose(c[row], expected, atol=1e-7):
                        raise ValueError("Centroid differs from corrected track")
                    validated += 1
    invalid_mother_edges = 0
    for i, tr in enumerate(tracks, start=1):
        mother = np.atleast_1d(tr["M"])
        if mother.size:
            mid = int(mother[0])
            if not 1 <= mid <= len(tracks):
                raise ValueError("Invalid mother track ID")
            if int(np.atleast_1d(tracks[mid - 1]["times"])[-1]) >= int(np.atleast_1d(tr["times"])[0]):
                # A stale manually corrected edge cannot certify ancestry
                # before this child track. Retain later within-track labels.
                tr["_invalid_mother"] = True
                invalid_mother_edges += 1
    audit = {"colony": path.stem.replace("fast_", ""), "n_frames": len(frames),
             "n_tracks": len(tracks), "n_detections": sum(map(len, frames.values())),
             "n_mapped": validated, "n_unmapped": unmapped,
             "n_daughter_tracks": sum(np.atleast_1d(tr["M"]).size > 0 for tr in tracks),
             "n_stale_age_indices": age_index_mismatches,
             "n_invalid_mother_edges": invalid_mother_edges,
             "n_extra_gold_observations": extra_gold_observations}
    return frames, mappings, tracks, audit


def ancestor_track(target_tid, source_frame, tracks):
    tid = int(target_tid)
    seen = set()
    while tid not in seen:
        seen.add(tid)
        tr = tracks[tid - 1]
        times = np.atleast_1d(tr["times"])
        if source_frame in times:
            return tid
        if source_frame > int(times[-1]):
            return None
        if tr.get("_invalid_mother", False):
            return None
        mother = np.atleast_1d(tr["M"])
        if mother.size == 0:
            return None
        tid = int(mother[0])
    raise ValueError("Cycle in mother links")


def make_window(source_frame, gap, frames, mappings, tracks, rng):
    target_frame = source_frame + gap
    src_order = rng.permutation(len(frames[source_frame]))
    dst_order = rng.permutation(len(frames[target_frame]))
    source = frames[source_frame][src_order]
    target = frames[target_frame][dst_order]
    src_mapping = mappings[source_frame][src_order]
    dst_mapping = mappings[target_frame][dst_order]
    src_index = {int(row[0]): i for i, row in enumerate(src_mapping)
                 if np.isfinite(row[0])}
    valid_rows, truth = [], []
    for j, row in enumerate(dst_mapping):
        if not np.isfinite(row[0]):
            continue
        ancestor = ancestor_track(row[0], source_frame, tracks)
        if ancestor in src_index:
            valid_rows.append(j)
            truth.append(src_index[ancestor])
    anchor_df = pd.DataFrame({"local_detection_id": np.arange(len(source)),
                              "center_y": source[:, 1], "center_x": source[:, 0]})
    target_df = pd.DataFrame({"local_detection_id": np.arange(len(target)),
                              "center_y": target[:, 1], "center_x": target[:, 0]})
    a = anchor_df[["center_y", "center_x"]].to_numpy()
    b = target_df[["center_y", "center_x"]].to_numpy()
    cost = np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=-1)
    e = {"anchor": anchor_df, "target": target_df,
         "window": {"horizon_frames": gap * MINUTES_PER_FRAME},
         "cost": cost.astype(np.float32)}
    return e, np.asarray(valid_rows, dtype=int), np.asarray(truth, dtype=int)


def family_f1(truth, predicted):
    keys = set(truth.tolist()) | set(predicted.tolist())
    return float(np.mean([2 * np.sum((truth == k) & (predicted == k)) /
                          (np.sum(truth == k) + np.sum(predicted == k))
                          for k in keys]))


def evaluate_colony(path, center_model, pair_model, rng):
    frames, mappings, tracks, audit = load_colony(path)
    print("audit", audit, flush=True)
    rows = []
    for gap in GAPS:
        for source_frame in sorted(frames):
            if source_frame + gap not in frames:
                continue
            e, valid_rows, truth = make_window(source_frame, gap, frames, mappings,
                                               tracks, rng)
            if len(truth) == 0:
                continue
            assignments = {"nearest_anchor": e["cost"].argmin(axis=1),
                           "family_center": multi_cost(e, center_model).argmin(axis=1),
                           "pairwise_geometry": pair_assignment(e, pair_model)}
            for method, assignment in assignments.items():
                guess = assignment[valid_rows]
                rows.append({"colony": audit["colony"],
                             "window_id": f"{source_frame}_{source_frame+gap}",
                             "source_frame": source_frame, "target_frame": source_frame + gap,
                             "gap_frames": gap, "gap_minutes_assumed": gap*MINUTES_PER_FRAME,
                             "n_anchors": len(e["anchor"]), "n_targets": len(e["target"]),
                             "n_scored": len(truth),
                             "n_excluded": len(e["target"]) - len(truth),
                             "growth_ratio": len(e["target"]) / len(e["anchor"]),
                             "method": method,
                             "accuracy": float(np.mean(guess == truth)),
                             "set_f1": family_f1(truth, guess)})
    return pd.DataFrame(rows), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    download_dir = ROOT / "data/downloads"
    if args.audit_only:
        path = download_dir / f"fast_{DEV}.zip"
        _, _, _, audit = load_colony(path)
        print("parser-development audit", audit)
        return
    paths = [download_dir / f"fast_{name}.zip" for name in HELD_OUT]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in HORIZONS
             for e in examples(alias, windows, compensate=False, horizon=h)]
    pair_model = train_classifier(train)
    center_model = fit_multi(train)
    results, audits = [], []
    rng = np.random.default_rng(20261002)
    for path in paths:
        one, audit = evaluate_colony(path, center_model, pair_model, rng)
        results.append(one)
        audits.append(audit)
        print("completed", audit["colony"], "windows", len(one)//3, flush=True)
    out = pd.concat(results, ignore_index=True)
    path = RESULTS / "fast_microcolony_external_test.parquet"
    out.to_parquet(path, index=False)
    pd.DataFrame(audits).to_csv(RESULTS / "fast_microcolony_external_audit.csv", index=False)
    summary = out.groupby(["colony", "gap_frames", "method"])[["accuracy", "set_f1"]].mean()
    print("per-colony", summary.round(4).to_string(), flush=True)
    print("equal-colony macro", summary.groupby(["gap_frames", "method"])
          .mean().round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
