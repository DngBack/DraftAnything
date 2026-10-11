"""Estimate family centroids from anonymous endpoint transport only.

Unlike oracle_family_flow_ceiling, no lineage or GT label is used to construct
the OT plan or centroids. Model affinity and decoder remain frozen. M3 is used
to select a small configuration grid; M4 is a viewed development movie.

Run from celltracking/: python experiments/ot_family_centroid_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import expansion_compensate  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from cellanc.transport import log_plan_from_cost  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402
from experiments.structure_flow_pilot import median_spacing  # noqa: E402


CONFIGS = (
    ("raw", 0.5, 3.0, 0.0),
    ("raw", 1.0, 3.0, 0.0),
    ("raw", 1.0, 3.0, 2.0),
    ("raw", 1.0, 10.0, 2.0),
    ("rc", 1.0, 3.0, 0.0),
    ("rc", 1.0, 3.0, 2.0),
)


def centroid_cost(e: dict, *, init: str, eps: float, tau: float, smooth: float) -> np.ndarray:
    src = e["anchor"][["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
    dst = e["target"][["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
    adjusted = dst if init == "raw" else expansion_compensate(src, dst)
    spacing = median_spacing(src)
    cost = ((src[:, None, 1:] - adjusted[None, :, 1:]) ** 2).sum(-1) / max(spacing**2, 1e-6)
    lp = log_plan_from_cost(cost, np.ones(len(src)), np.ones(len(dst)), eps, tau,
                            iters=120, tol=1e-3)
    conditional = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
    centroids = conditional @ dst[:, 1:]
    if smooth:
        displacement = centroids - src[:, 1:]
        k = min(32, len(src))
        dist, near = cKDTree(src[:, 1:]).query(src[:, 1:], k=k)
        dist = np.atleast_2d(dist).reshape(len(src), k)
        near = np.atleast_2d(near).reshape(len(src), k)
        weight = np.exp(-0.5 * (dist / (smooth * spacing)) ** 2)
        weight /= weight.sum(axis=1, keepdims=True).clip(1e-12)
        centroids = src[:, 1:] + (weight[..., None] * displacement[near]).sum(axis=1)
    return (((dst[:, None, 1:] - centroids[None, :, :]) ** 2).sum(-1) /
            max(spacing**2, 1e-6)).astype(np.float32)


def metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    return ((df.n_correct / df.n_scored).mean(),
            (df.set_f1_sum / df.count_n_anchors).mean(),
            df.n_correct.sum() / df.n_scored.sum())


def evaluate(ex: list[dict], decoder: dict, cfg: tuple) -> pd.DataFrame:
    init, eps, tau, smooth = cfg
    name = f"ot_center_{init}_e{eps:g}_t{tau:g}_h{smooth:g}"
    changed = [{**e, "cost": centroid_cost(e, init=init, eps=eps, tau=tau,
                                           smooth=smooth)} for e in ex]
    out = summarize(changed, **decoder).assign(method=name)
    out["n_anchors"] = [len(e["anchor"]) for e in ex]
    return out


def main():
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    model = Affinity()
    model.load_state_dict(saved["model"])
    model.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    val = examples("M3", windows, compensate=False)
    probabilities(model, saved["mean"], saved["std"], val)
    base_val = summarize(val, **decoder).assign(method="family_raw")
    base_val["n_anchors"] = [len(e["anchor"]) for e in val]
    rows = [base_val]
    ranked = []
    for cfg in CONFIGS:
        scored = evaluate(val, decoder, cfg)
        rows.append(scored)
        ma, f1, pooled = metrics(scored)
        dense = scored[scored.n_anchors > 200]
        dma, df1, _ = metrics(dense)
        ranked.append((cfg, ma, f1, pooled, dma, df1))
        print("M3", cfg, tuple(round(v, 4) for v in (ma, f1, pooled, dma, df1)), flush=True)
    # Select on M3, giving equal priority to dense-window accuracy and F1.
    chosen = max(ranked, key=lambda x: x[4] + x[5])[0]
    print("M3 selected", chosen, flush=True)
    test = examples("M4", windows, compensate=False)
    probabilities(model, saved["mean"], saved["std"], test)
    base_test = summarize(test, **decoder).assign(method="family_raw")
    base_test["n_anchors"] = [len(e["anchor"]) for e in test]
    rows.extend([base_test, evaluate(test, decoder, chosen)])
    data = pd.concat(rows, ignore_index=True)
    path = RESULTS / "ot_family_centroid_h132.parquet"
    data.to_parquet(path, index=False)
    (RESULTS / "ot_family_centroid_h132_config.json").write_text(json.dumps({
        "selected_on_M3": {"init": chosen[0], "eps": chosen[1],
                            "tau": chosen[2], "smooth": chosen[3]},
        "selection": "maximize M3 dense macro accuracy + dense set-F1",
        "M4_status": "viewed development movie, not fresh held-out test",
    }, indent=2))
    for seq in ("M3", "M4"):
        subset = data[data.sequence_id == seq]
        for method, df in subset.groupby("method"):
            print(seq, method, "all", tuple(round(v, 4) for v in metrics(df)),
                  "dense", tuple(round(v, 4) for v in metrics(df[df.n_anchors > 200])),
                  flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
