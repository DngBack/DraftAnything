"""Proliferation-conditioned center prior and zero-shot CTC transfer probe.

Train on OIAM M0-M2 endpoint gaps 1..132 minutes. Compare physical-time and
observed-growth clocks with identical regressors/features. Evaluate M3/M4
with the frozen H=132 family decoder; evaluate CTC only by nearest center.
This is exploratory and CTC windows are few; no CTC labels train the model.

Run from celltracking/: python experiments/growth_clock_transfer.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.family import endpoint_cost  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.ctc_center_transfer import select_windows  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402
from experiments.learned_family_center_probe import geometry  # noqa: E402
from experiments.multihorizon_center_context import SHIFT  # noqa: E402
from experiments.oracle_family_flow_ceiling import family_centroids  # noqa: E402


TRAIN_H = (1, 2, 4, 8, 16, 33, 66, 132)


def feature(e, clock):
    x, _, spacing, src = geometry(e)
    if clock == "physical_time":
        value = np.log2(float(e["window"]["horizon_min"]) / 132)
    else:
        value = np.log2(max(len(e["target"]) / len(e["anchor"]), 0.1))
    return np.column_stack([x[:, SHIFT], np.full((len(src), 1), value)]), spacing, src


def fit(items, clock):
    xs, ys = [], []
    for e in items:
        x, spacing, src = feature(e, clock)
        keep, truth = family_centroids(e)
        xs.append(x[keep])
        ys.append((truth - src[keep]) / spacing)
    x, y = np.concatenate(xs), np.concatenate(ys)
    print(clock, "training", x.shape, y.shape, flush=True)
    return [HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05,
            max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1,
            random_state=0).fit(x, y[:, j]) for j in range(2)]


def predicted_cost(e, models, clock):
    x, spacing, src = feature(e, clock)
    center = src + spacing * np.column_stack([m.predict(x) for m in models])
    target = e["target"][["center_y", "center_x"]].to_numpy(dtype=float)
    return (((target[:, None, :] - center[None, :, :]) ** 2).sum(-1) /
            max(spacing**2, 1e-6)).astype(np.float32)


def evaluate_oiam(windows, models):
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    affinity = Affinity()
    affinity.load_state_dict(saved["model"])
    affinity.eval()
    rows = []
    for alias in ("M3", "M4"):
        for h in (33, 66, 132):
            items = examples(alias, windows, compensate=False, horizon=h)
            probabilities(affinity, saved["mean"], saved["std"], items)
            for name in ("family_raw", *models):
                data = items if name == "family_raw" else [
                    {**e, "cost": predicted_cost(e, models[name], name)} for e in items]
                part = summarize(data, **decoder).assign(method=name, horizon_frames=h)
                part["n_anchors"] = [len(e["anchor"]) for e in items]
                rows.append(part)
            print("OIAM", alias, h, "complete", flush=True)
    out = pd.concat(rows, ignore_index=True)
    path = RESULTS / "growth_clock_oiam.parquet"
    out.to_parquet(path, index=False)
    for label, data in (("all", out), ("dense", out[out.n_anchors > 200])):
        acc = (data.n_correct / data.n_scored).groupby([data.horizon_frames, data.method]).mean()
        f1 = (data.set_f1_sum / data.count_n_anchors).groupby([data.horizon_frames, data.method]).mean()
        print(label, pd.DataFrame({"acc": acc, "f1": f1}).round(4).to_string(), flush=True)
    print("saved", path, flush=True)


def evaluate_ctc(windows, models):
    selected = select_windows(windows)
    empty = pd.DataFrame(columns=["det_1", "det_2"])
    rows = []
    for alias in sorted({w["sequence_id"] for w in selected}):
        det = pd.read_parquet(BENCH / "inputs/detections" / f"{alias}.parquet")
        by_frame = {int(t): g.sort_values("local_detection_id").reset_index(drop=True)
                    for t, g in det.groupby("frame_index")}
        targets = dict(list(pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
                            .groupby("window_id")))
        for w in (w for w in selected if w["sequence_id"] == alias):
            anchor, target = by_frame[w["anchor_frame"]], by_frame[w["target_frame"]]
            if not len(anchor) or not len(target):
                continue
            e = {"anchor": anchor, "target": target, "window": w}
            costs = {"nearest_anchor": endpoint_cost(anchor, target, compensate=False)}
            costs.update({name: predicted_cost(e, model, name) for name, model in models.items()})
            for name, c in costs.items():
                ids = anchor.local_detection_id.to_numpy(dtype=int)[c.argmin(axis=1)]
                pred = dict(zip(target.local_detection_id.to_numpy(dtype=int), ids))
                s = score_window(pred, set(), targets[w["window_id"]], empty)
                rows.append({"sequence_id": alias, "window_id": w["window_id"],
                             "horizon_min": w["horizon_min"], "method": name,
                             "n_anchors": len(anchor), "n_scored": s["n_scored"],
                             "n_correct": s["n_correct"], "set_f1_sum": s["set_f1_sum"],
                             "count_n_anchors": s["count_n_anchors"]})
        print("CTC", alias, "complete", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "growth_clock_ctc.parquet"
    out.to_parquet(path, index=False)
    out = out[out.n_scored > 0].copy()
    out["acc"] = out.n_correct / out.n_scored
    out["f1"] = out.set_f1_sum / out.count_n_anchors
    print(out.groupby(["sequence_id", "method"])[["acc", "f1"]].mean().round(4).to_string())
    print("saved", path, flush=True)


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in TRAIN_H
             for e in examples(alias, windows, compensate=False, horizon=h,
                               require_experiment=h in (33, 66, 132))]
    models = {name: fit(train, name) for name in ("physical_time", "growth_clock")}
    evaluate_oiam(windows, models)
    evaluate_ctc(windows, models)


if __name__ == "__main__":
    main()
