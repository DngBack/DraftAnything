"""Supervised family-center registration using observed endpoint point clouds.

Training labels are descendant centroids on M0-M2. Inference features use only
anchor and target locations from the two retained frames. Affinity and decoder
are frozen. M3 selects a small regressor grid; M4 was viewed previously.

Run from celltracking/: python experiments/learned_family_center_probe.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402
from experiments.oracle_family_flow_ceiling import family_centroids  # noqa: E402


def geometry(e: dict) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    src = e["anchor"][["center_y", "center_x"]].to_numpy(dtype=float)
    dst = e["target"][["center_y", "center_x"]].to_numpy(dtype=float)
    n, m = len(src), len(dst)
    ca, cb = np.median(src, axis=0), np.median(dst, axis=0)
    rs = np.linalg.norm(src - ca, axis=1)
    rt = np.linalg.norm(dst - cb, axis=1)
    scale_a = max(np.percentile(rs, 80), 1.0)
    scale_b = max(np.percentile(rt, 80), 1.0)
    if n > 1:
        ds, _ = cKDTree(src).query(src, k=min(n, 9))
        ds = np.asarray(ds).reshape(n, -1)
        spacing = max(float(np.median(ds[:, 1])), 1.0)
    else:
        ds = np.zeros((n, 1))
        spacing = 1.0
    k = min(m, 8)
    dt, ind = cKDTree(dst).query(src, k=k)
    dt = np.asarray(dt).reshape(n, k)
    ind = np.asarray(ind).reshape(n, k)
    local4 = dst[ind[:, :min(4, k)]].mean(axis=1)
    local8 = dst[ind].mean(axis=1)
    spread = np.sqrt(np.mean(np.sum((dst[ind] - local8[:, None, :]) ** 2, axis=2), axis=1))

    cols = [
        (src - ca) / scale_a,
        (src - cb) / scale_b,
        (local4 - src) / spacing,
        (local8 - src) / spacing,
        ((cb - ca) / spacing)[None, :].repeat(n, axis=0),
        (rs / scale_a)[:, None],
        np.full((n, 1), np.log1p(n)),
        np.full((n, 1), np.log1p(m)),
        np.full((n, 1), m / max(n, 1)),
        np.full((n, 1), scale_b / scale_a),
        np.full((n, 1), spacing / scale_a),
        (ds[:, [min(j, ds.shape[1] - 1) for j in (1, 3, 7)]] / spacing),
        (dt[:, [min(j, k - 1) for j in (0, 3, 7)]] / spacing),
        (spread / spacing)[:, None],
    ]
    return np.concatenate(cols, axis=1).astype(np.float32), local4, spacing, src


def regression_data(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    for e in items:
        x, local, spacing, _ = geometry(e)
        keep, centers = family_centroids(e)
        features.append(x[keep])
        targets.append(((centers - local[keep]) / spacing).astype(np.float32))
    return np.concatenate(features), np.concatenate(targets)


def reg_cost(e: dict, models=None) -> np.ndarray:
    x, local, spacing, src = geometry(e)
    dst = e["target"][["center_y", "center_x"]].to_numpy(dtype=float)
    if models is None:
        centers = local
    else:
        residual = np.column_stack([m.predict(x) for m in models])
        centers = local + spacing * residual
    return (((dst[:, None, :] - centers[None, :, :]) ** 2).sum(-1) /
            max(spacing**2, 1e-6)).astype(np.float32)


def score(ex: list[dict], decoder: dict, name: str, models=None) -> pd.DataFrame:
    costs = [{**e, "cost": reg_cost(e, models)} for e in ex]
    out = summarize(costs, **decoder).assign(method=name)
    out["n_anchors"] = [len(e["anchor"]) for e in ex]
    return out


def metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    return ((df.n_correct / df.n_scored).mean(),
            (df.set_f1_sum / df.count_n_anchors).mean(),
            df.n_correct.sum() / df.n_scored.sum())


def main():
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    affinity = Affinity()
    affinity.load_state_dict(saved["model"])
    affinity.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for e in examples(alias, windows, compensate=False)]
    x, y = regression_data(train)
    print("train", x.shape, y.shape, "target RMSE", np.sqrt(np.mean(y**2, axis=0)), flush=True)
    val = examples("M3", windows, compensate=False)
    probabilities(affinity, saved["mean"], saved["std"], val)
    rows = [score(val, decoder, "local4"), score(val, decoder, "family_raw")]
    # The original endpoint cost is anchor-centered, unlike the local4 control.
    rows[-1] = summarize(val, **decoder).assign(method="family_raw")
    rows[-1]["n_anchors"] = [len(e["anchor"]) for e in val]
    candidates = []
    for leaves in (15, 31, 63):
        models = [HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05,
                    max_leaf_nodes=leaves, min_samples_leaf=30, l2_regularization=1,
                    random_state=0).fit(x, y[:, j]) for j in range(2)]
        name = f"learned_center_l{leaves}"
        out = score(val, decoder, name, models)
        rows.append(out)
        ma, f1, pool = metrics(out)
        dense_ma, dense_f1, _ = metrics(out[out.n_anchors > 200])
        candidates.append((leaves, ma, f1, pool, dense_ma, dense_f1, models))
        print("M3", name, tuple(round(float(v), 4) for v in (ma, f1, pool, dense_ma, dense_f1)), flush=True)
    selected = max(candidates, key=lambda c: c[4] + c[5])
    print("selected on M3 dense", selected[0], flush=True)
    test = examples("M4", windows, compensate=False)
    probabilities(affinity, saved["mean"], saved["std"], test)
    base = summarize(test, **decoder).assign(method="family_raw")
    base["n_anchors"] = [len(e["anchor"]) for e in test]
    rows.extend([base, score(test, decoder, "local4"),
                 score(test, decoder, f"learned_center_l{selected[0]}", selected[6])])
    data = pd.concat(rows, ignore_index=True)
    path = RESULTS / "learned_family_center_h132.parquet"
    data.to_parquet(path, index=False)
    joblib.dump({"models": selected[6], "features": x.shape[1], "train_aliases": ("M0", "M1", "M2"),
                 "selection_alias": "M3", "selected_leaves": selected[0]},
                RESULTS / "learned_family_center_h132.joblib")
    for seq in ("M3", "M4"):
        subset = data[data.sequence_id == seq]
        for name, df in subset.groupby("method"):
            print(seq, name, "all", tuple(round(float(v), 4) for v in metrics(df)),
                  "dense", tuple(round(float(v), 4) for v in metrics(df[df.n_anchors > 200])),
                  flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
