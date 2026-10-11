"""Post-hoc mechanism audit: errors by *true* endpoint family size.

Family size is evaluator-only and must never become an inference feature.
OIAM M3/M4 and seven FAST colonies have already been viewed. This analysis
is exploratory; it is intended to test whether a proposed family-size
mechanism is consistent with errors, not to certify a new model.

Run from celltracking/: python experiments/family_size_error_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.fast_microcolony_external_test import (GAPS, HELD_OUT,
                                                        load_colony, make_window)  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402
from experiments.pairwise_geometry_baseline import (HORIZONS, pair_assignment,
                                                    train_classifier)  # noqa: E402


def family_bin(counts):
    return np.select([counts == 1, counts == 2, counts <= 4, counts <= 8],
                     ["1", "2", "3-4", "5-8"], default="9+")


def rows_for_window(dataset, sequence, horizon, window_id, e, valid, truth,
                    center_model, pair_model):
    if len(truth) == 0:
        return []
    sizes = np.bincount(truth, minlength=len(e["anchor"]))[truth]
    bins = family_bin(sizes)
    assignments = {"nearest_anchor": e["cost"].argmin(axis=1),
                   "family_center": multi_cost(e, center_model).argmin(axis=1),
                   "pairwise_geometry": pair_assignment(e, pair_model)}
    output = []
    for name, prediction in assignments.items():
        correct = prediction[valid] == truth
        for label in np.unique(bins):
            keep = bins == label
            output.append({"dataset": dataset, "sequence": sequence,
                           "horizon_min": horizon, "window_id": window_id,
                           "n_anchors": len(e["anchor"]),
                           "growth_ratio": len(e["target"]) / len(e["anchor"]),
                           "family_size_bin": label, "method": name,
                           "n_targets": int(keep.sum()),
                           "n_correct": int(correct[keep].sum())})
    return output


def oiam_truth(e):
    source = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
    target = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
    rows = e["targets"]
    rows = rows[rows.in_foi & (rows.target_status == "valid_anchor")]
    valid = []
    truth = []
    for _, row in rows.iterrows():
        j = target.get(int(row.target_detection_id))
        i = source.get(int(row.ancestor_detection_id))
        if j is not None and i is not None:
            valid.append(j)
            truth.append(i)
    return np.asarray(valid, dtype=int), np.asarray(truth, dtype=int)


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in HORIZONS
             for e in examples(alias, windows, compensate=False, horizon=h)]
    pair_model = train_classifier(train)
    center_model = fit_multi(train)
    results = []
    for alias in ("M3", "M4"):
        for gap in HORIZONS:
            for e in examples(alias, windows, compensate=False, horizon=gap):
                valid, truth = oiam_truth(e)
                results.extend(rows_for_window("OIAM", alias, gap,
                               e["window"]["window_id"], e, valid, truth,
                               center_model, pair_model))
            print("OIAM", alias, gap, "complete", flush=True)
    rng = np.random.default_rng(20261002)
    for alias in HELD_OUT:
        path = ROOT / "data/downloads" / f"fast_{alias}.zip"
        frames, mappings, tracks, audit = load_colony(path)
        for gap in GAPS:
            for source_frame in sorted(frames):
                if source_frame + gap not in frames:
                    continue
                e, valid, truth = make_window(source_frame, gap, frames,
                                               mappings, tracks, rng)
                results.extend(rows_for_window("FAST", alias, gap * 5,
                               f"{source_frame}_{source_frame+gap}", e,
                               valid, truth, center_model, pair_model))
        print("FAST", alias, "complete", flush=True)
    out = pd.DataFrame(results)
    path = RESULTS / "family_size_error_analysis.parquet"
    out.to_parquet(path, index=False)
    summary = out.groupby(["dataset", "horizon_min", "family_size_bin", "method"])[
        ["n_correct", "n_targets"]].sum()
    summary["accuracy"] = summary.n_correct / summary.n_targets
    print(summary[["n_targets", "accuracy"]].round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
