"""Does family-center prediction use observed target-cloud evidence?

Train fixed-capacity regressors on M0-M2 with source-only, source+global-target,
and full local-target geometry. The family affinity and decoder are frozen.
M4 is previously viewed development data. No GT enters inference features.

Run from celltracking/: python experiments/family_center_information_ablation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402
from experiments.learned_family_center_probe import geometry  # noqa: E402
from experiments.oracle_family_flow_ceiling import family_centroids  # noqa: E402


SOURCE = [0, 1, 10, 11, 15, 16, 17, 18]
GLOBAL = sorted(SOURCE + [2, 3, 8, 9, 12, 13, 14])
LOCAL = list(range(23))
VARIANTS = {
    "source_prior": (SOURCE, "source"),
    "target_shift": (sorted(SOURCE + [2, 3, 8, 9]), "source"),
    "target_growth": (sorted(SOURCE + [12, 13, 14]), "source"),
    "global_context": (GLOBAL, "source"),
    "local_context": (LOCAL, "source"),
    "local4_residual": (LOCAL, "local4"),
}


def data(items: list[dict], indices: list[int], base: str) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = [], []
    for e in items:
        x, local4, spacing, src = geometry(e)
        keep, truth = family_centroids(e)
        origin = src if base == "source" else local4
        xx.append(x[keep][:, indices])
        yy.append((truth - origin[keep]) / spacing)
    return np.concatenate(xx), np.concatenate(yy)


def fit(items: list[dict], indices: list[int], base: str):
    x, y = data(items, indices, base)
    return [HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05,
            max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1,
            random_state=0).fit(x, y[:, j]) for j in range(2)]


def cost(e: dict, indices: list[int], base: str, models) -> np.ndarray:
    x, local4, spacing, src = geometry(e)
    origin = src if base == "source" else local4
    center = origin + spacing * np.column_stack([m.predict(x[:, indices]) for m in models])
    target = e["target"][["center_y", "center_x"]].to_numpy(dtype=float)
    return (((target[:, None, :] - center[None, :, :]) ** 2).sum(-1) /
            max(spacing**2, 1e-6)).astype(np.float32)


def main():
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    affinity = Affinity()
    affinity.load_state_dict(saved["model"])
    affinity.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for e in examples(alias, windows, compensate=False)]
    models = {name: fit(train, idx, base) for name, (idx, base) in VARIANTS.items()}
    rows = []
    for alias in ("M3", "M4"):
        items = examples(alias, windows, compensate=False)
        probabilities(affinity, saved["mean"], saved["std"], items)
        baseline = summarize(items, **decoder).assign(method="family_raw")
        baseline["n_anchors"] = [len(e["anchor"]) for e in items]
        rows.append(baseline)
        for name, (idx, base) in VARIANTS.items():
            changed = [{**e, "cost": cost(e, idx, base, models[name])} for e in items]
            result = summarize(changed, **decoder).assign(method=name)
            result["n_anchors"] = [len(e["anchor"]) for e in items]
            rows.append(result)
        print(alias, "complete", flush=True)
    out = pd.concat(rows, ignore_index=True)
    path = RESULTS / "family_center_information_ablation_h132.parquet"
    out.to_parquet(path, index=False)
    out["acc"] = out.n_correct / out.n_scored
    out["f1"] = out.set_f1_sum / out.count_n_anchors
    print("All windows")
    print(out.groupby(["sequence_id", "method"])[["acc", "f1"]].mean().round(4).to_string())
    print("Dense >200 anchors")
    print(out[out.n_anchors > 200].groupby(["sequence_id", "method"])[["acc", "f1"]].mean().round(4).to_string())
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
