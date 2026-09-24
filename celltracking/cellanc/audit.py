"""Index + audit one CTC sequence (doc §5–6). Produces tables, never edits raw data."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import tifffile

from .ctc import Sequence, marker_centroids


def _tif_meta(path):
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        return tuple(p.shape), str(p.dtype)


def build_index(seq: Sequence, workers: int = 8):
    """Return (frames, tracks, observations) DataFrames with GT track IDs.

    observations keeps gt_track_id: it is the evaluator-side table. Model inputs must
    be relabelled (see relabel_frame) before use (doc §3).
    """
    images, markers = seq.images(), seq.markers()
    frame_ids = sorted(set(images) | set(markers))
    with ProcessPoolExecutor(workers) as ex:
        img_meta = dict(zip(images, ex.map(_tif_meta, images.values(), chunksize=16)))
        mk_meta = dict(zip(markers, ex.map(_tif_meta, markers.values(), chunksize=16)))
        cents = dict(zip(markers, ex.map(marker_centroids, markers.values(), chunksize=16)))

    frames = pd.DataFrame([{
        "dataset": seq.dataset, "sequence_id": seq.seq, "frame_index": t,
        "image_path": str(images[t]) if t in images else None,
        "shape": img_meta[t][0] if t in img_meta else None,
        "dtype": img_meta[t][1] if t in img_meta else None,
        "marker_path": str(markers[t]) if t in markers else None,
        "marker_shape": mk_meta[t][0] if t in mk_meta else None,
        "marker_dtype": mk_meta[t][1] if t in mk_meta else None,
    } for t in frame_ids])

    tracks = pd.DataFrame(
        [(seq.dataset, seq.seq, L, B, E, P) for L, (B, E, P) in seq.tracks().items()],
        columns=["dataset", "sequence_id", "gt_track_id", "start_frame", "end_frame", "parent_id"])

    observations = pd.DataFrame(
        [(seq.dataset, seq.seq, t, L, cy, cx, n)
         for t, c in cents.items() for L, (cy, cx, n) in c.items()],
        columns=["dataset", "sequence_id", "frame_index", "gt_track_id",
                 "center_y", "center_x", "marker_pixels"])
    return frames, tracks, observations


def audit(frames: pd.DataFrame, tracks: pd.DataFrame, obs: pd.DataFrame):
    """Checks 1–6 of doc §6. Returns (summary_row, issues DataFrame)."""
    issues = []

    def flag(kind, **kw):
        issues.append({"issue": kind, **kw})

    # 2. image/marker shape + dtype
    for r in frames.itertuples():
        if r.image_path is None:
            flag("marker_without_image", frame=r.frame_index)
        elif r.marker_path is not None and r.shape != r.marker_shape:
            flag("shape_mismatch", frame=r.frame_index)
        if r.marker_dtype and not r.marker_dtype.startswith("uint"):
            flag("marker_dtype_not_uint", frame=r.frame_index, dtype=r.marker_dtype)

    T = {r.gt_track_id: (r.start_frame, r.end_frame, r.parent_id) for r in tracks.itertuples()}
    # 3. start<=end, parent refs, cycles
    for L, (B, E, P) in T.items():
        if B > E:
            flag("start_after_end", track=L)
        if P != 0 and P not in T:
            flag("parent_missing", track=L, parent=P)
    for L in T:
        seen, u = set(), L
        while u in T and T[u][2] != 0:
            if u in seen:
                flag("cycle", track=L)
                break
            seen.add(u)
            u = T[u][2]
    # 4. parent ends before child starts; record gaps
    for L, (B, E, P) in T.items():
        if P in T:
            gap = B - T[P][1] - 1
            if gap < 0:
                flag("child_overlaps_parent", track=L, parent=P)
            elif gap > 0:
                flag("parent_child_gap", track=L, parent=P, gap=gap)
    # 5. children per parent
    kids = Counter(P for (_, _, P) in T.values() if P != 0)
    for P, n in kids.items():
        if n != 2:
            flag("children_not_2", track=P, n_children=n)
    # 2 + 6. marker IDs known, inside [B,E], and present at every frame of [B,E]
    marker_frames = set(frames.loc[frames.marker_path.notna(), "frame_index"])
    seen_at = obs.groupby("gt_track_id")["frame_index"].apply(set).to_dict()
    for L, fs in seen_at.items():
        if L not in T:
            flag("marker_id_not_in_table", track=L)
            continue
        B, E, _ = T[L]
        if outside := [t for t in fs if not B <= t <= E]:
            flag("marker_outside_span", track=L, n=len(outside))
    for L, (B, E, _) in T.items():
        expected = {t for t in range(B, E + 1) if t in marker_frames}
        if missing := expected - seen_at.get(L, set()):
            flag("missing_marker_in_span", track=L, n=len(missing))

    iss = pd.DataFrame(issues) if issues else pd.DataFrame(columns=["issue"])
    counts = iss.issue.value_counts().to_dict() if len(iss) else {}
    summary = {
        "dataset": frames.dataset.iat[0], "sequence_id": frames.sequence_id.iat[0],
        "n_frames": int(frames.image_path.notna().sum()),
        "n_marker_frames": len(marker_frames),
        "shapes": sorted({str(s) for s in frames["shape"].dropna()}),
        "dtypes": sorted(frames.dtype.dropna().unique().tolist()),
        "n_observations": len(obs),
        "n_tracks": len(T),
        "n_root_tracks": sum(P == 0 for (_, _, P) in T.values()),
        "n_binary_divisions": sum(n == 2 for n in kids.values()),
        **{f"issue_{k}": v for k, v in counts.items()},
    }
    return summary, iss
