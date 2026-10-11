"""Observed-cloud registration probes for sparse endpoint ancestry.

All maps use only anchor and target centers in the two retained frames. The
saved family affinity and decoder are frozen. M4 is a viewed development movie.

Run from celltracking/: python experiments/unsupervised_registration_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import expansion_compensate  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402
from experiments.structure_flow_pilot import anonymous_flow, median_spacing  # noqa: E402


def quantile_values(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = (rankdata(target, method="average") - 0.5) / len(target)
    q = (np.arange(len(source)) + 0.5) / len(source)
    return np.interp(p, q, np.sort(source))


def map_points(src: np.ndarray, dst: np.ndarray, method: str) -> np.ndarray:
    out = dst.astype(float).copy()
    a, b = src[:, 1:], dst[:, 1:]
    if method == "raw":
        return out
    if method == "rc":
        return expansion_compensate(src, dst)
    if method == "axis_quantile":
        for j in range(2):
            out[:, j + 1] = quantile_values(a[:, j], b[:, j])
        return out
    ca, cb = np.median(a, axis=0), np.median(b, axis=0)
    if method == "axis_iqr":
        sa = np.subtract(*np.percentile(a, [90, 10], axis=0))
        sb = np.subtract(*np.percentile(b, [90, 10], axis=0))
        out[:, 1:] = ca + (b - cb) * (sa / sb.clip(1e-6))
        return out
    if method == "radial_quantile":
        ra, rb = np.linalg.norm(a - ca, axis=1), np.linalg.norm(b - cb, axis=1)
        r = quantile_values(ra, rb)
        out[:, 1:] = ca + (b - cb) * (r / rb.clip(1e-6))[:, None]
        return out
    if method == "axis_quantile_flow":
        prepared = map_points(src, dst, "axis_quantile")
        return anonymous_flow(src, prepared, bandwidth=2, strength=0.5,
                              initialize="raw")
    if method == "radial_quantile_flow":
        prepared = map_points(src, dst, "radial_quantile")
        return anonymous_flow(src, prepared, bandwidth=2, strength=0.5,
                              initialize="raw")
    raise ValueError(method)


def reg_cost(e: dict, method: str) -> np.ndarray:
    a = e["anchor"][["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
    b = e["target"][["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
    mapped = map_points(a, b, method)
    spacing = median_spacing(a)
    return (((mapped[:, None, 1:] - a[None, :, 1:]) ** 2).sum(-1) /
            max(spacing ** 2, 1e-6)).astype(np.float32)


def main():
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    model = Affinity()
    model.load_state_dict(saved["model"])
    model.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    methods = ("raw", "rc", "axis_iqr", "axis_quantile", "radial_quantile",
               "axis_quantile_flow", "radial_quantile_flow")
    rows = []
    for alias in ("M3", "M4"):
        ex = examples(alias, windows, compensate=False)
        probabilities(model, saved["mean"], saved["std"], ex)
        for method in methods:
            changed = [{**e, "cost": reg_cost(e, method)} for e in ex]
            result = summarize(changed, **decoder).assign(method=method)
            result["n_anchors"] = [len(e["anchor"]) for e in ex]
            rows.append(result)
            print(alias, method, "complete", flush=True)
    data = pd.concat(rows, ignore_index=True)
    path = RESULTS / "unsupervised_registration_h132.parquet"
    data.to_parquet(path, index=False)
    data["acc"] = data.n_correct / data.n_scored
    data["f1"] = data.set_f1_sum / data.count_n_anchors
    for title, subset in (("All windows", data), ("Dense >200 anchors", data[data.n_anchors > 200])):
        print(title)
        print(subset.groupby(["sequence_id", "method"])[["acc", "f1"]].mean().round(4).to_string())
    print("saved", path)


if __name__ == "__main__":
    main()
