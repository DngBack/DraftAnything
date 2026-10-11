"""Exploratory direct pairwise/family-center rank ensemble and safety rule.

All evaluation collections were viewed before this probe. A positive result
here is development evidence only; recA colonies are reserved for a later
prospective test after the method is frozen.

Run from celltracking/: python experiments/hybrid_rank_probe.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.ambiguity_regime_analysis import ambiguity  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.fast_microcolony_external_test import (GAPS as FAST_GAPS,
                                                        HELD_OUT, family_f1,
                                                        load_colony, make_window)  # noqa: E402
from experiments.lfct_example_external_test import (XML, load_graph,
                                                    make_window as lfct_window)  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402
from experiments.pairwise_geometry_baseline import (HORIZONS, pair_probabilities,
                                                    score, train_classifier)  # noqa: E402
from experiments.tui_t98g_external_test import (GAPS as T98G_GAPS, MASK_DIR,
                                                frame_table, lineage, target_labels)  # noqa: E402


def ranks(cost, descending=False):
    order = np.argsort(-cost if descending else cost, axis=1)
    result = np.empty_like(order)
    result[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])
    return result


def assignments(e, center_model, pair_model):
    center_cost = multi_cost(e, center_model)
    pair_prob = pair_probabilities(e, pair_model)
    cr = ranks(center_cost)
    pr = ranks(pair_prob, descending=True)
    n = len(e["anchor"])
    w = float(np.clip(np.log(max(n, 1) / 16) / np.log(200 / 16), 0, 1))
    raw = e["cost"].argmin(axis=1)
    mixed = ((1 - w) * pr + w * cr).argmin(axis=1)
    source = e["anchor"][["center_y", "center_x"]].to_numpy()
    target = e["target"][["center_y", "center_x"]].to_numpy()
    obs = ambiguity(source, target)
    safe_raw = obs["growth_ratio"] < 1.1 and obs["nearest_distance_scaled"] < 0.6
    return {"nearest_anchor": raw, "pairwise_geometry": pair_prob.argmax(axis=1),
            "family_center": center_cost.argmin(axis=1),
            "hybrid_half": (0.5 * pr + 0.5 * cr).argmin(axis=1),
            "hybrid_density": mixed,
            "hybrid_safe": raw if safe_raw else mixed}


def external_rows(dataset, sequence, window_id, horizon, e, valid, truth,
                  center_model, pair_model):
    if len(truth) == 0:
        return []
    rows = []
    for name, pred in assignments(e, center_model, pair_model).items():
        guess = pred[valid]
        rows.append({"dataset": dataset, "sequence": sequence, "window_id": window_id,
                     "horizon_min": horizon, "n_anchors": len(e["anchor"]),
                     "n_targets": len(e["target"]), "n_scored": len(truth),
                     "method": name, "accuracy": float(np.mean(guess == truth)),
                     "set_f1": family_f1(truth, guess)})
    return rows


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in HORIZONS
             for e in examples(alias, windows, compensate=False, horizon=h)]
    pair_model = train_classifier(train)
    center_model = fit_multi(train)
    rows = []
    for alias in ("M3", "M4"):
        for horizon in HORIZONS:
            for e in examples(alias, windows, compensate=False, horizon=horizon):
                for name, assignment in assignments(e, center_model, pair_model).items():
                    result = score(e, assignment)
                    rows.append({"dataset": "OIAM", "sequence": alias,
                                 "window_id": e["window"]["window_id"],
                                 "horizon_min": horizon, "n_anchors": len(e["anchor"]),
                                 "n_targets": len(e["target"]), "n_scored": result["n_scored"],
                                 "method": name,
                                 "accuracy": result["n_correct"] / result["n_scored"],
                                 "set_f1": result["set_f1_sum"] / result["count_n_anchors"]})
            print("OIAM", alias, horizon, "done", flush=True)
    rng = np.random.default_rng(20261002)
    for alias in HELD_OUT:
        frames, mappings, tracks, _ = load_colony(ROOT / "data/downloads" /
                                                   f"fast_{alias}.zip")
        for gap in FAST_GAPS:
            for source in sorted(frames):
                if source + gap not in frames:
                    continue
                e, valid, truth = make_window(source, gap, frames, mappings,
                                               tracks, rng)
                rows.extend(external_rows("FAST", alias, f"{source}_{source+gap}",
                                            gap * 5, e, valid, truth,
                                            center_model, pair_model))
        print("FAST", alias, "done", flush=True)
    spots, by_frame, parent = load_graph(XML)
    for gap in (2, 4, 8):
        for source in sorted(by_frame):
            if source + gap not in by_frame:
                continue
            e, valid, truth = lfct_window(source, gap, spots, by_frame, parent, rng)
            rows.extend(external_rows("LFCT", "FLD_3", f"{source}_{source+gap}",
                                        gap * 15, e, valid, truth,
                                        center_model, pair_model))
        print("LFCT", gap, "done", flush=True)
    parent = lineage(MASK_DIR / "man_track.txt")
    frames = {t: frame_table(MASK_DIR / f"mask{t:03}.tif", t) for t in range(37)}
    for _, gap in T98G_GAPS.items():
        for source in range(37 - gap):
            anchor, source_ids = frames[source]
            target, target_ids = frames[source + gap]
            labels, _ = target_labels(source_ids, target_ids, parent)
            valid_labels = labels[labels.target_status == "valid_anchor"]
            valid = valid_labels.target_detection_id.to_numpy(dtype=int) - 1
            truth = valid_labels.ancestor_detection_id.to_numpy(dtype=int) - 1
            a = anchor[["center_y", "center_x"]].to_numpy()
            b = target[["center_y", "center_x"]].to_numpy()
            cost = np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=-1)
            e = {"anchor": anchor, "target": target,
                 "window": {"horizon_frames": gap * 10},
                 "cost": cost.astype(np.float32)}
            rows.extend(external_rows("T98G", "T98G", f"{source}_{source+gap}",
                                        gap * 10, e, valid, truth,
                                        center_model, pair_model))
        print("T98G", gap, "done", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "hybrid_rank_probe.parquet"
    out.to_parquet(path, index=False)
    summary = out.groupby(["dataset", "horizon_min", "method"])[
        ["accuracy", "set_f1"]].mean()
    print(summary.round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
