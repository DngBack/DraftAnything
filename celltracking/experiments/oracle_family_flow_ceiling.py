"""Diagnostic only: how much would better family-scale registration help?

GT ancestry determines descendant centroids and therefore enters the cost.
These are deliberately oracle conditions, never valid inference results.

Run from celltracking/: python experiments/oracle_family_flow_ceiling.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402
from experiments.structure_flow_pilot import median_spacing  # noqa: E402


def family_centroids(e: dict) -> tuple[np.ndarray, np.ndarray]:
    target = e["target"]
    ids = target.local_detection_id.to_numpy(dtype=int)
    pos = target[["center_y", "center_x"]].to_numpy(dtype=float)
    pos_by_id = dict(zip(ids, pos))
    tg = e["targets"]
    valid = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
    centroids = valid.groupby("ancestor_detection_id").target_detection_id.apply(
        lambda children: np.mean([pos_by_id[int(i)] for i in children], axis=0))
    aid = e["anchor"].local_detection_id.to_numpy(dtype=int)
    keep = np.array([i in centroids.index for i in aid])
    return keep, np.stack([centroids[i] for i in aid[keep]])


def cost_oracle(e: dict, bandwidth: float | None) -> np.ndarray:
    anchor = e["anchor"][["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
    target = e["target"][["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
    spacing = median_spacing(anchor)
    keep, centroids = family_centroids(e)
    if bandwidth is None:
        # Family centroids are a strongly leaked upper-bound diagnostic: their
        # locations were calculated using the answer labels.
        cost = ((target[:, None, 1:] - anchor[None, :, 1:]) ** 2).sum(-1)
        cost[:, keep] = ((target[:, None, 1:] - centroids[None, :, :]) ** 2).sum(-1)
    else:
        source_pos = anchor[keep, 1:]
        displacement = source_pos - centroids
        k = min(16, len(centroids))
        d, idx = cKDTree(centroids).query(target[:, 1:], k=k)
        d = np.atleast_2d(d).reshape(len(target), k)
        idx = np.atleast_2d(idx).reshape(len(target), k)
        weight = np.exp(-0.5 * (d / max(bandwidth * spacing, 1e-6)) ** 2)
        weight /= weight.sum(axis=1, keepdims=True).clip(1e-12)
        mapped = target[:, 1:] + (weight[..., None] * displacement[idx]).sum(axis=1)
        cost = ((mapped[:, None, :] - anchor[None, :, 1:]) ** 2).sum(-1)
    return (cost / max(spacing ** 2, 1e-6)).astype(np.float32)


def main():
    config = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    model = Affinity()
    model.load_state_dict(saved["model"])
    model.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    output = []
    for alias in ("M3", "M4"):
        ex = examples(alias, windows, compensate=False)
        probabilities(model, saved["mean"], saved["std"], ex)
        variants = {"raw": [e["cost"] for e in ex]}
        for bw in (1.0, 2.0, 4.0, 8.0):
            variants[f"oracle_smooth_h{bw:g}"] = [cost_oracle(e, bw) for e in ex]
        variants["oracle_family_centroid"] = [cost_oracle(e, None) for e in ex]
        for name, costs in variants.items():
            changed = [{**e, "cost": cost} for e, cost in zip(ex, costs, strict=True)]
            scores = summarize(changed, **config).assign(method=name)
            scores["n_anchors"] = [len(e["anchor"]) for e in ex]
            output.append(scores)
        print(alias, "complete", flush=True)
    data = pd.concat(output, ignore_index=True)
    path = RESULTS / "oracle_family_flow_ceiling_h132.parquet"
    data.to_parquet(path, index=False)
    data["acc"] = data.n_correct / data.n_scored
    data["f1"] = data.set_f1_sum / data.count_n_anchors
    print(data.groupby(["sequence_id", "method"])[["acc", "f1"]].mean().round(4).to_string())
    print("Dense >200 anchors:")
    print(data[data.n_anchors > 200].groupby(["sequence_id", "method"])[["acc", "f1"]].mean().round(4).to_string())
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
