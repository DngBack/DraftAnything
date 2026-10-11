"""Horizon-conditioned family-center registration, M0-M2 to M3/M4.

Training windows at H=33,66,132 overlap within each movie. M4 has been viewed
in prior pilots. The affinity and decoder are frozen H=132 models so this probes
the motion prior only; it is not a standalone multi-horizon tracker benchmark.

Run from celltracking/: python experiments/multihorizon_center_context.py
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
from experiments.family_center_information_ablation import SOURCE, cost, fit  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402
from experiments.learned_family_center_probe import geometry  # noqa: E402
from experiments.oracle_family_flow_ceiling import family_centroids  # noqa: E402


SHIFT = sorted(SOURCE + [2, 3, 8, 9])


def features(e, indices=SHIFT):
    x, _, spacing, src = geometry(e)
    horizon = e["window"]["horizon_frames"]
    t = np.full((len(src), 1), np.log2(horizon / 132), dtype=np.float32)
    return np.column_stack([x[:, indices], t]), spacing, src


def fit_multi(items, include_horizon=True, indices=SHIFT):
    xs, ys = [], []
    for e in items:
        x, spacing, src = features(e, indices)
        keep, truth = family_centroids(e)
        xs.append(x[keep] if include_horizon else x[keep, :-1])
        ys.append((truth - src[keep]) / spacing)
    x, y = np.concatenate(xs), np.concatenate(ys)
    print("training", x.shape, y.shape, flush=True)
    return [HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05,
            max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1,
            random_state=0).fit(x, y[:, j]) for j in range(2)]


def multi_cost(e, models, include_horizon=True, indices=SHIFT):
    x, spacing, src = features(e, indices)
    if not include_horizon:
        x = x[:, :-1]
    center = src + spacing * np.column_stack([m.predict(x) for m in models])
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
    train132 = [e for alias in ("M0", "M1", "M2")
                for e in examples(alias, windows, compensate=False, horizon=132)]
    trainall = [e for alias in ("M0", "M1", "M2") for h in (33, 66, 132)
                for e in examples(alias, windows, compensate=False, horizon=h)]
    single = fit(train132, SHIFT, "source")
    multi = fit_multi(trainall)
    pooled = fit_multi(trainall, include_horizon=False)
    rows = []
    for alias in ("M3", "M4"):
        for horizon in (33, 66, 132):
            items = examples(alias, windows, compensate=False, horizon=horizon)
            probabilities(affinity, saved["mean"], saved["std"], items)
            for name in ("family_raw", "single_h132", "pooled_no_horizon", "multi_horizon"):
                if name == "family_raw":
                    data = items
                else:
                    data = [{**e, "cost": (cost(e, SHIFT, "source", single)
                                            if name == "single_h132" else
                                            multi_cost(e, pooled, False)
                                            if name == "pooled_no_horizon" else
                                            multi_cost(e, multi))}
                            for e in items]
                part = summarize(data, **decoder).assign(method=name, horizon_frames=horizon)
                part["n_anchors"] = [len(e["anchor"]) for e in items]
                rows.append(part)
            print(alias, horizon, "complete", flush=True)
    out = pd.concat(rows, ignore_index=True)
    path = RESULTS / "multihorizon_center_context.parquet"
    out.to_parquet(path, index=False)
    for label, data in (("all", out), ("dense", out[out.n_anchors > 200])):
        acc = (data.n_correct / data.n_scored).groupby([data.horizon_frames, data.method]).mean()
        f1 = (data.set_f1_sum / data.count_n_anchors).groupby([data.horizon_frames, data.method]).mean()
        print(label, pd.DataFrame({"acc": acc, "f1": f1}).round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
