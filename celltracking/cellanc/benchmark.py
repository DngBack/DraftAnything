"""Build benchmark v0: relabelled inputs, windows, evaluator labels (doc §3, §7–11).

Layout under <root> = outputs/benchmarks/v0:
  inputs/detections/<alias>.parquet   frame_index, local_detection_id, center_y, center_x
  inputs/frames/<alias>.parquet       frame_index, timestamp_min, image_path (relative to data/)
  inputs/windows/<alias>.jsonl        one record per (window, schedule); no GT fields
  labels/gt_mapping/<alias>.parquet   frame_index, local_detection_id, gt_track_id, match_status
  labels/targets/<alias>.parquet      per window x target cell at b
  labels/siblings/<alias>.parquet     direct sister pairs at b
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .lineage import (ancestry_targets, hidden_intermediates, n_children, schedule,
                      sibling_pairs)


@dataclass
class SeqData:
    dataset: str
    sequence_id: str
    alias: str
    split: str
    time_step_min: float
    foi_margin_px: float
    shape: tuple[int, int]
    obs: pd.DataFrame  # evaluator-side observations (with gt_track_id)
    tracks: dict  # gt id -> (start, end, parent)
    frames: pd.DataFrame


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def relabel(obs: pd.DataFrame, alias: str, seed: int) -> pd.DataFrame:
    """Per-frame seeded permutation -> local_detection_id (doc §3). Output sorted by local id."""
    parts = []
    for t, g in obs.groupby("frame_index", sort=True):
        g = g.copy()
        g["local_detection_id"] = np.random.default_rng(_seed(seed, alias, t)).permutation(len(g)) + 1
        parts.append(g)
    return pd.concat(parts).sort_values(["frame_index", "local_detection_id"]).reset_index(drop=True)


def cell_cycle_frames(tracks: dict) -> np.ndarray:
    """Durations (frames, inclusive) of tracks that both start and end with a real division.

    Single-child links (CTC gap closing) are not divisions and are excluded on both ends.
    """
    kids = n_children(tracks)
    return np.array([e - s + 1 for L, (s, e, p) in tracks.items()
                     if p and kids[p] >= 2 and kids[L] >= 2])


def schedules_for(a: int, b: int, rng_seed: int) -> dict[str, list[int]]:
    """Experiment A schedules on a fixed (a, b): dense, power-of-2 strides, endpoints, random."""
    H = b - a
    out = {f"s{s}": schedule(a, b, s) for s in [2**k for k in range(12)] if s < H}
    out["ends"] = [a, b]
    rng = np.random.default_rng(rng_seed)
    for k in (1, 3):
        if H - 1 >= k:
            interior = sorted(rng.choice(np.arange(a + 1, b), size=k, replace=False).tolist())
            out[f"rand{k}"] = [a, *interior, b]
    return out


def build_sequence(sd: SeqData, horizons: list[int], a_step: int, min_anchors: int,
                   deltas: list[int], seed: int):
    """Return (local_obs, windows, targets, siblings) for one sequence."""
    local = relabel(sd.obs, sd.alias, seed)
    to_local = dict(zip(zip(local.frame_index, local.gt_track_id), local.local_detection_id))
    pos = dict(zip(zip(local.frame_index, local.gt_track_id), zip(local.center_y, local.center_x)))
    present = local.groupby("frame_index").gt_track_id.apply(set).to_dict()
    last = max(present)
    H_img, W_img = sd.shape
    m = sd.foi_margin_px

    def in_foi(y, x):
        return m <= y < H_img - m and m <= x < W_img - m

    anchors_ok = [t for t in sorted(present) if len(present[t]) >= min_anchors]
    a_grid = list(range(anchors_ok[0], last, a_step)) if anchors_ok else []

    specs = {}  # (a, b, schedule_id) -> (frames, experiments)
    for a in a_grid:
        for H in horizons:
            if a + H <= last:
                for sid, fr in schedules_for(a, a + H, _seed(seed, sd.alias, a, a + H)).items():
                    specs.setdefault((a, a + H, sid), [fr, set()])[1].add(f"A_h{H}")
        for d in deltas:
            if a + d <= last:
                specs.setdefault((a, a + d, "ends"), [[a, a + d], set()])[1].add("B")

    windows, targets, sibs = [], [], []
    trace_cache = {}
    for (a, b, sid), (fr, exps) in sorted(specs.items()):
        wid = f"{sd.alias}_a{a:04d}_b{b:04d}_{sid}"
        windows.append({
            "window_id": wid, "split": sd.split, "dataset": sd.dataset, "sequence_id": sd.alias,
            "anchor_frame": a, "target_frame": b, "horizon_frames": b - a,
            "horizon_min": (b - a) * sd.time_step_min, "schedule_id": sid,
            "observed_frames": fr, "timestamps_min": [t * sd.time_step_min for t in fr],
            "experiments": sorted(exps), "detections_source": "oracle_locations_relabelled",
            "task": "ancestor_at_reference_time", "seed": seed,
        })
        if (a, b) not in trace_cache:
            trace_cache[(a, b)] = ancestry_targets(a, b, sd.tracks, present, [a, b])
        kept = set(fr)
        for r in trace_cache[(a, b)]:
            u, anc = r["target_gt"], r["anchor_gt"]
            targets.append({
                "window_id": wid,
                "target_detection_id": to_local[(b, u)],
                "ancestor_detection_id": to_local[(a, anc)] if anc else pd.NA,
                "target_status": r["status"], "generation_depth": r["generation_depth"],
                "hidden_intermediates": (hidden_intermediates(u, anc, sd.tracks, kept)
                                         if anc else pd.NA),
                "in_foi": in_foi(*pos[(b, u)]),
            })
        for x, y in sibling_pairs(b, sd.tracks, present[b]):
            sibs.append({"window_id": wid, "det_1": to_local[(b, x)], "det_2": to_local[(b, y)]})

    targets = pd.DataFrame(targets).astype({"ancestor_detection_id": "Int64",
                                            "hidden_intermediates": "Int64"})
    sibs = pd.DataFrame(sibs, columns=["window_id", "det_1", "det_2"])
    return local, windows, targets, sibs


def write_sequence(root, data_dir, sd: SeqData, local, windows, targets, sibs):
    for d in ["inputs/detections", "inputs/frames", "inputs/windows", "labels/gt_mapping",
              "labels/targets", "labels/siblings"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    local[["frame_index", "local_detection_id", "center_y", "center_x"]].to_parquet(
        root / "inputs/detections" / f"{sd.alias}.parquet", index=False)
    fr = sd.frames[sd.frames.image_path.notna()]
    pd.DataFrame({
        "frame_index": fr.frame_index,
        "timestamp_min": fr.frame_index * sd.time_step_min,
        "image_path": [str(Path(p).resolve().relative_to(data_dir.resolve()))
                       for p in fr.image_path],
    }).to_parquet(root / "inputs/frames" / f"{sd.alias}.parquet", index=False)
    local.assign(match_status="oracle")[["frame_index", "local_detection_id", "gt_track_id",
                                         "match_status"]].to_parquet(
        root / "labels/gt_mapping" / f"{sd.alias}.parquet", index=False)
    targets.to_parquet(root / "labels/targets" / f"{sd.alias}.parquet", index=False)
    sibs.to_parquet(root / "labels/siblings" / f"{sd.alias}.parquet", index=False)
    with open(root / "inputs/windows" / f"{sd.alias}.jsonl", "w") as f:
        for w in windows:
            f.write(json.dumps(w) + "\n")
