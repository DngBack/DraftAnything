"""Can cloud geometry reject a shifted motion prior without test labels?

Exploratory after viewing T98G results: compare raw and physical-time center
costs by a symmetric nearest-neighbor cloud energy, then choose one per window.
Training remains OIAM M0-M2 only. OIAM M3/M4 and T98G are development data.

Run from celltracking/: python experiments/unsupervised_prior_gate_probe.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.family import endpoint_cost  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.ctc_center_transfer import select_windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.growth_clock_transfer import TRAIN_H, fit, predicted_cost  # noqa: E402
from experiments.tui_t98g_external_test import (GAPS, MASK_DIR, frame_table,
                                               lineage, target_labels)  # noqa: E402


def cloud_energy(cost):
    """Symmetric Chamfer energy after normalizing by anchor spacing."""
    return float(cost.min(axis=1).mean() + cost.min(axis=0).mean())


def evaluate(e, truth, model, empty):
    raw = endpoint_cost(e["anchor"], e["target"], compensate=False)
    learned = predicted_cost(e, model, "physical_time")
    energies = {"raw": cloud_energy(raw), "learned": cloud_energy(learned)}
    scores = {}
    for name, cost in (("raw", raw), ("learned", learned)):
        anchor_ids = e["anchor"].local_detection_id.to_numpy(dtype=int)
        target_ids = e["target"].local_detection_id.to_numpy(dtype=int)
        pred = dict(zip(target_ids, anchor_ids[cost.argmin(axis=1)]))
        s = score_window(pred, set(), truth, empty)
        scores[name] = s["n_correct"] / s["n_scored"]
    choice = min(energies, key=energies.get)
    oracle = max(scores, key=scores.get)
    return {"raw_acc": scores["raw"], "learned_acc": scores["learned"],
            "gate_acc": scores[choice], "oracle_acc": scores[oracle],
            "gate_choice": choice, "energy_raw": energies["raw"],
            "energy_learned": energies["learned"],
            "n_anchors": len(e["anchor"]), "n_targets": len(e["target"])}


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in TRAIN_H
             for e in examples(alias, windows, compensate=False, horizon=h,
                               require_experiment=h in (33, 66, 132))]
    model = fit(train, "physical_time")
    empty = pd.DataFrame(columns=["det_1", "det_2"])
    rows = []
    for alias in ("M3", "M4"):
        for h in (33, 66, 132):
            items = examples(alias, windows, compensate=False, horizon=h)
            for e in items:
                rows.append({"dataset": "OIAM", "sequence_id": alias,
                             "window_id": e["window"]["window_id"], "horizon_min": h,
                             **evaluate(e, e["targets"], model, empty)})
    parent = lineage(MASK_DIR / "man_track.txt")
    frames = {t: frame_table(MASK_DIR / f"mask{t:03}.tif", t) for t in range(37)}
    for intended, gap in GAPS.items():
        for a in range(37 - gap):
            b = a + gap
            src, src_gt = frames[a]
            dst, dst_gt = frames[b]
            truth, _ = target_labels(src_gt, dst_gt, parent)
            e = {"anchor": src, "target": dst,
                 "window": {"horizon_min": gap * 10}}
            rows.append({"dataset": "T98G", "sequence_id": "T98G",
                         "window_id": f"T98G_a{a:03}_b{b:03}", "horizon_min": gap * 10,
                         **evaluate(e, truth, model, empty)})
    selected = select_windows(windows)
    for alias in sorted({w["sequence_id"] for w in selected}):
        det = pd.read_parquet(BENCH / "inputs/detections" / f"{alias}.parquet")
        by_frame = {int(t): g.sort_values("local_detection_id").reset_index(drop=True)
                    for t, g in det.groupby("frame_index")}
        labels = dict(list(pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
                           .groupby("window_id")))
        for w in (w for w in selected if w["sequence_id"] == alias):
            src, dst = by_frame[w["anchor_frame"]], by_frame[w["target_frame"]]
            if not len(src) or not len(dst):
                continue
            e = {"anchor": src, "target": dst, "window": w}
            rows.append({"dataset": w["dataset"], "sequence_id": alias,
                         "window_id": w["window_id"], "horizon_min": w["horizon_min"],
                         **evaluate(e, labels[w["window_id"]], model, empty)})
    out = pd.DataFrame(rows)
    path = RESULTS / "unsupervised_prior_gate_probe.parquet"
    out.to_parquet(path, index=False)
    print(out.groupby(["dataset", "horizon_min"])[["raw_acc", "learned_acc", "gate_acc", "oracle_acc"]].mean().round(4).to_string(), flush=True)
    print(out.groupby(["dataset", "horizon_min", "gate_choice"]).size().to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
