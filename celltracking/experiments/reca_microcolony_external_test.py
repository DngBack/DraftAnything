"""Prospective OIAM -> nine recA microcolony endpoint ancestry evaluation.

Protocol: docs/261002_reca_future_holdout_protocol.md. Training uses only
OIAM M0-M2. No recA labels enter inference. Each frame is independently
relabelled. The recA parser was developed on 151027-05 only, which is never
included in this test result.

Run from celltracking/: python experiments/reca_microcolony_external_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.fast_microcolony_external_test import family_f1  # noqa: E402
from experiments.hybrid_rank_probe import assignments  # noqa: E402
from experiments.multihorizon_center_context import fit_multi  # noqa: E402
from experiments.pairwise_geometry_baseline import HORIZONS, train_classifier  # noqa: E402
from experiments.reca_schnitzcells import ancestor_track, load_colony  # noqa: E402

HELD_OUT = ("151027-06", "151027-10", "151028-01", "151028-08",
            "151029-05", "151029-09", "151029-11", "151031-03",
            "151031-13")
GAPS = (4, 8, 16)
MINUTES_PER_FRAME = 7.5


def make_window(start, gap, frames, mapping, tracks, invalid_parents, rng):
    end = start + gap
    src_order = rng.permutation(len(frames[start]))
    dst_order = rng.permutation(len(frames[end]))
    src_xy = frames[start][src_order]
    dst_xy = frames[end][dst_order]
    src_ids = mapping[start][src_order]
    dst_ids = mapping[end][dst_order]
    src_index = {int(tid): i for i, tid in enumerate(src_ids) if np.isfinite(tid)}
    valid, truth = [], []
    for j, tid in enumerate(dst_ids):
        if not np.isfinite(tid):
            continue
        ancestor = ancestor_track(tid, start, tracks, invalid_parents)
        if ancestor in src_index:
            valid.append(j)
            truth.append(src_index[ancestor])
    anchor = pd.DataFrame({"local_detection_id": np.arange(len(src_xy)),
                           "center_y": src_xy[:, 1], "center_x": src_xy[:, 0]})
    target = pd.DataFrame({"local_detection_id": np.arange(len(dst_xy)),
                           "center_y": dst_xy[:, 1], "center_x": dst_xy[:, 0]})
    cost = np.sum((dst_xy[:, None, :] - src_xy[None, :, :]) ** 2, axis=-1)
    e = {"anchor": anchor, "target": target,
         "window": {"horizon_frames": gap * MINUTES_PER_FRAME},
         "cost": cost.astype(np.float32)}
    return e, np.asarray(valid, dtype=int), np.asarray(truth, dtype=int)


def main():
    roots = [ROOT / "data/raw/reca_holdout" / name for name in HELD_OUT]
    missing = [str(r) for r in roots if not (r / "data" / f"{r.name}_lin.mat").exists()]
    if missing:
        raise FileNotFoundError(f"recA holdout files absent: {missing}")
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in HORIZONS
             for e in examples(alias, windows, compensate=False, horizon=h)]
    pair_model = train_classifier(train)
    center_model = fit_multi(train)
    rng = np.random.default_rng(20261002)
    all_rows, audits = [], []
    for root in roots:
        frames, mapping, tracks, invalid_parents, audit = load_colony(root)
        audits.append(audit)
        print("audit", audit, flush=True)
        for gap in GAPS:
            for start in sorted(frames):
                if start + gap not in frames or len(frames[start]) == 0 or len(frames[start+gap]) == 0:
                    continue
                e, valid, truth = make_window(start, gap, frames, mapping,
                                               tracks, invalid_parents, rng)
                if len(truth) == 0:
                    continue
                for method, assignment in assignments(e, center_model, pair_model).items():
                    guess = assignment[valid]
                    all_rows.append({"colony": root.name, "window_id": f"{start}_{start+gap}",
                                     "source_frame": start, "target_frame": start + gap,
                                     "gap_frames": gap, "gap_minutes": gap * MINUTES_PER_FRAME,
                                     "n_anchors": len(e["anchor"]),
                                     "n_targets": len(e["target"]), "n_scored": len(truth),
                                     "n_excluded": len(e["target"]) - len(truth),
                                     "growth_ratio": len(e["target"]) / len(e["anchor"]),
                                     "method": method,
                                     "accuracy": float(np.mean(guess == truth)),
                                     "set_f1": family_f1(truth, guess)})
            print(root.name, gap, "complete", flush=True)
    data = pd.DataFrame(all_rows)
    path = RESULTS / "reca_microcolony_external_test.parquet"
    data.to_parquet(path, index=False)
    pd.DataFrame(audits).to_csv(RESULTS / "reca_microcolony_external_audit.csv", index=False)
    per_colony = data.groupby(["colony", "gap_frames", "method"])[
        ["accuracy", "set_f1"]].mean()
    print("equal-colony macro", per_colony.groupby(["gap_frames", "method"])
          .mean().round(4).to_string(), flush=True)
    print("per-colony", per_colony.round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
