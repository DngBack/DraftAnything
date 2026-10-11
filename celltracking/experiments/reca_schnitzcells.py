"""Schnitzcells recA centroid and lineage adapter; no model inference.

Use the parser-development colony 151027-05 before opening the nine reserved
colonies. Coordinates from each per-frame segmentation feature file are the
input detections. Schnitzcells track IDs and P mother links are evaluator-only.
"""

import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def _centroids(path):
    try:
        with h5py.File(path, "r") as f:
            raw = np.asarray(f["cellSegData/cellCentroid"][()])
            declared = int(np.asarray(f["cellSegData/cellNum"][()]).ravel()[0])
            rect = np.asarray(f["cellSegData/rect"][()]).ravel()
    except OSError:
        data = scipy.io.loadmat(path, simplify_cells=True)["cellSegData"]
        raw = np.asarray(data["cellCentroid"])
        declared = int(data["cellNum"])
        rect = np.asarray(data["rect"]).ravel()
    if raw.ndim != 2:
        raise ValueError(f"Invalid centroids in {path}")
    if raw.shape[0] == 2:
        xy = raw.T
    elif raw.shape[1] == 2:
        xy = raw
    else:
        raise ValueError(f"Invalid centroid shape in {path}: {raw.shape}")
    if len(xy) != declared:
        raise ValueError(f"Invalid centroid count in {path}")
    if len(rect) != 4 or not np.all(np.isfinite(rect)) or rect[2] < rect[0] or rect[3] < rect[1]:
        raise ValueError(f"Invalid crop rectangle in {path}: {rect}")
    # Schnitzcells centroids are local to the per-frame crop. MATLAB's rect
    # order is [row_min, col_min, row_max, col_max], while centroids are x/y.
    # Convert its one-based origin to x/y coordinates of the original image.
    return xy, rect[[1, 0]] - 1


def _as_int_array(value):
    a = np.atleast_1d(value)
    return a.astype(int) if a.size else np.empty(0, dtype=int)


def _tracks(path):
    try:
        raw = scipy.io.loadmat(path, simplify_cells=True)["schnitzcells"]
        return list(raw) if isinstance(raw, (list, np.ndarray)) else [raw]
    except NotImplementedError:
        # The same recA archive mixes MATLAB v5 and v7.3 lineage files.
        # v7.3 stores each struct field as HDF5 object references.
        with h5py.File(path, "r") as f:
            group = f["schnitzcells"]
            fields = ("frames", "cellno", "cenx", "ceny", "P")
            n_tracks = group["frames"].size
            if any(group[field].size != n_tracks for field in fields):
                raise ValueError("Schnitzcells fields have different track counts")
            refs = {field: group[field][()].ravel() for field in fields}
            return [{field: np.asarray(f[refs[field][i]][()]).ravel()
                     for field in fields}
                    for i in range(n_tracks)]


def load_colony(root, skip_bad_rect=False):
    root = Path(root)
    mats = sorted((root / "cellSegmentationData").glob("*_segData*.mat"))
    if not mats:
        raise FileNotFoundError(f"No segmentation features in {root}")
    by_index = {}
    origins = {}
    for path in mats:
        match = re.search(r"_segData(\d+)\.mat$", path.name)
        if not match:
            raise ValueError(f"Unexpected segmentation filename: {path}")
        index = int(match.group(1))
        try:
            by_index[index], origins[index] = _centroids(path)
        except ValueError as err:
            # blind test 4 amendment: frames whose crop rectangle is stored empty are dropped
            if not (skip_bad_rect and "Invalid crop rectangle" in str(err)):
                raise
            print("dropped frame", path.name, flush=True)
    lineage_path = root / "data" / f"{root.name}_lin.mat"
    tracks = _tracks(lineage_path)
    # Determine the frame-number offset by exact centroid agreement; the
    # development archive uses segData001 for Schnitzcells frame 2.
    candidates = []
    for offset in range(-2, 4):
        matched = checked = 0
        for tr in tracks:
            frames = _as_int_array(tr["frames"])
            cellnos = _as_int_array(tr["cellno"])
            xs = np.atleast_1d(tr["cenx"])
            ys = np.atleast_1d(tr["ceny"])
            if not (len(frames) == len(cellnos) == len(xs) == len(ys)):
                raise ValueError("Schnitzcells track arrays have different lengths")
            for frame, cellno, x, y in zip(frames, cellnos, xs, ys):
                a = by_index.get(frame - offset)
                if a is None or cellno < 1 or cellno > len(a):
                    continue
                if not np.all(np.isfinite(a[cellno - 1])):
                    continue
                checked += 1
                matched += bool(np.allclose(a[cellno - 1], [x, y], atol=1e-6))
        candidates.append((matched, checked, offset))
    candidates.sort(reverse=True)
    matched, checked, offset = candidates[0]
    if checked == 0 or matched / checked < 0.99:
        raise ValueError(f"Could not align lineage centroids with segmentation: {candidates}")
    if len(candidates) > 1 and candidates[1][0] == matched:
        raise ValueError("Ambiguous frame-number offset")
    valid_indices = {i + offset: np.flatnonzero(np.isfinite(xy).all(axis=1))
                     for i, xy in by_index.items()}
    frames = {i + offset: xy[valid_indices[i + offset]]
              for i, xy in by_index.items()}
    lookup = {f: {int(old): new for new, old in enumerate(ids)}
              for f, ids in valid_indices.items()}
    mapping = {f: np.full(len(xy), np.nan) for f, xy in frames.items()}
    matched_observations = 0
    missing_observations = 0
    mismatched_observations = 0
    for tid, tr in enumerate(tracks, start=1):
        for frame, cellno, x, y in zip(_as_int_array(tr["frames"]),
                                       _as_int_array(tr["cellno"]),
                                       np.atleast_1d(tr["cenx"]),
                                       np.atleast_1d(tr["ceny"])):
            if frame not in frames:
                missing_observations += 1
                continue
            original = by_index[frame - offset]
            if cellno < 1 or cellno > len(original):
                raise ValueError("Track cell number outside frame feature table")
            if cellno - 1 not in lookup[frame]:
                missing_observations += 1
                continue
            row = lookup[frame][cellno - 1]
            if not np.allclose(frames[frame][row], [x, y], atol=1e-6):
                # A few saved track coordinates refer to a different
                # segmentation object. Keep the detection as an unlabeled
                # input; exclude this mapping from ancestry scoring.
                mismatched_observations += 1
                continue
            if np.isfinite(mapping[frame][row]):
                raise ValueError("Two tracks map to one detection")
            mapping[frame][row] = tid
            matched_observations += 1
    for frame, row in mapping.items():
        ids = row[np.isfinite(row)].astype(int)
        if len(ids) != len(set(ids)):
            raise ValueError(f"One track maps to multiple detections in frame {frame}")
    invalid_parents = set()
    for tid, tr in enumerate(tracks, start=1):
        mother = _as_int_array(tr["P"])
        if mother.size == 0 or mother[0] < 1:
            continue
        mid = int(mother[0])
        if mid > len(tracks):
            invalid_parents.add(tid)
            continue
        parent_frames = _as_int_array(tracks[mid - 1]["frames"])
        child_frames = _as_int_array(tr["frames"])
        if not parent_frames.size or not child_frames.size or parent_frames[-1] >= child_frames[0]:
            invalid_parents.add(tid)
    audit = {"colony": root.name, "n_frames": len(frames), "frame_offset": offset,
             "n_tracks": len(tracks), "n_detections": sum(map(len, frames.values())),
             "n_mapped": matched_observations,
             "n_unmapped": sum(len(x) - np.isfinite(mapping[f]).sum()
                               for f, x in frames.items()),
             "n_invalid_centroids": sum(len(x) - len(frames[i + offset])
                                         for i, x in by_index.items()),
             "n_lineage_observations_outside_features": missing_observations,
             "n_mismatched_track_centroids": mismatched_observations,
             "n_invalid_parent_edges": len(invalid_parents)}
    frames = {f: xy + origins[f - offset] for f, xy in frames.items()}
    audit["coordinate_system"] = "original_image_via_rect"
    return frames, mapping, tracks, invalid_parents, audit


def ancestor_track(tid, source_frame, tracks, invalid_parents):
    tid = int(tid)
    seen = set()
    while tid not in seen:
        seen.add(tid)
        frames = _as_int_array(tracks[tid - 1]["frames"])
        if source_frame in frames:
            return tid
        if not frames.size or source_frame > frames[-1] or tid in invalid_parents:
            return None
        mother = _as_int_array(tracks[tid - 1]["P"])
        if mother.size == 0 or mother[0] < 1:
            return None
        tid = int(mother[0])
    raise ValueError("Cycle in Schnitzcells parent links")
