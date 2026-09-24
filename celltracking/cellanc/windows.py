"""Build ancestry windows: model inputs without GT IDs, evaluator labels apart (doc §3, §8–10)."""

import numpy as np
import pandas as pd

from .lineage import ancestry_targets, schedule, sibling_pairs


def relabel_frame(obs_t: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign per-frame local_detection_id by a seeded permutation; keeps gt_track_id for mapping."""
    rng = np.random.default_rng(seed)
    out = obs_t.reset_index(drop=True).copy()
    out["local_detection_id"] = rng.permutation(len(out)) + 1
    return out


def make_window(seq_obs: pd.DataFrame, tracks: dict, a: int, b: int, stride: int | None,
                time_step_min: float, seed: int, sequence_alias: str):
    """Return (input_record, gt_mapping rows, target rows, sibling rows)."""
    frames = schedule(a, b, stride)
    present = {t: set(g.gt_track_id) for t, g in seq_obs[seq_obs.frame_index.isin(frames)]
               .groupby("frame_index")}
    tag = f"{sequence_alias}_a{a:04d}_b{b:04d}_s{stride or 0}"
    local = {t: relabel_frame(seq_obs[seq_obs.frame_index == t], seed * 100003 + t) for t in frames}

    to_local = {(t, g): l for t, df in local.items()
                for g, l in zip(df.gt_track_id, df.local_detection_id)}
    inp = {
        "window_id": tag, "sequence_id": sequence_alias,
        "anchor_frame": a, "target_frame": b, "observed_frames": frames,
        "timestamps_min": [t * time_step_min for t in frames],
        "detections_source": "oracle_locations_relabelled",
        "task": "ancestor_at_reference_time", "schedule_stride": stride, "seed": seed,
        "detections": {str(t): [[int(l), float(y), float(x)] for l, y, x in df.sort_values(
                           "local_detection_id")[["local_detection_id", "center_y", "center_x"]]
                           .itertuples(index=False)] for t, df in local.items()},
    }
    mapping = [{"window_id": tag, "frame_index": t, "local_detection_id": int(l),
                "gt_track_id": int(g), "match_status": "oracle"} for (t, g), l in to_local.items()]
    targets = []
    for r in ancestry_targets(a, b, tracks, present, frames):
        targets.append({
            "window_id": tag,
            "target_detection_id": int(to_local[(b, r["target_gt"])]),
            "ancestor_detection_id": int(to_local[(a, r["anchor_gt"])]) if r["anchor_gt"] else None,
            "target_status": r["status"], "generation_depth": r["generation_depth"],
            "hidden_intermediates": r["hidden_intermediates"],
        })
    sibs = [{"window_id": tag, "det_1": int(to_local[(b, x)]), "det_2": int(to_local[(b, y)])}
            for x, y in sibling_pairs(b, tracks, present.get(b, set()))]
    return inp, mapping, targets, sibs
