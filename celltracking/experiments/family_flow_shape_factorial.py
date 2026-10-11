"""Frozen 2x2 ablation: anonymous flow x observable single-frame shape.

Uses the saved M0-M2 affinity model and M3-selected decoder/flow settings.
M3 and previously viewed M4 are descriptive development movies, not fresh
held-out tests. Shape permutation changes only per-target mask moments at the
observed endpoint. Ground truth is used for scoring, not flow or inference.

Run from celltracking/: python experiments/family_flow_shape_factorial.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_flow_joint_pilot import flow_cost  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402
from experiments.family_shape_ablation import shuffled_shape  # noqa: E402


def score_variant(items: list[dict], cost: list[np.ndarray], config: dict,
                  method: str) -> pd.DataFrame:
    with_cost = [{**e, "cost": c} for e, c in zip(items, cost, strict=True)]
    out = summarize(with_cost, **config)
    out["method"] = method
    out["n_anchors"] = [len(e["anchor"]) for e in items]
    return out


def summarize_group(df: pd.DataFrame) -> pd.Series:
    acc = df.n_correct / df.n_scored
    f1 = df.set_f1_sum / df.count_n_anchors
    return pd.Series({"n_windows": len(df), "macro_acc": acc.mean(),
                      "set_f1": f1.mean(),
                      "pooled_acc": df.n_correct.sum() / df.n_scored.sum()})


def main():
    cfg = json.loads((RESULTS / "family_flow_joint_h132_config.json").read_text())
    decoder = cfg["base_config"]
    flow = cfg["selected_flow"]
    assert flow == {"bandwidth": 2, "strength": 0.5}, "Unexpected changed flow choice"
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    assert tuple(saved["train_aliases"]) == ("M0", "M1", "M2")
    model = Affinity()
    model.load_state_dict(saved["model"])
    model.eval()

    all_windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        original = examples(alias, all_windows, compensate=False)
        probabilities(model, saved["mean"], saved["std"], original)
        raw_costs = [e["cost"] for e in original]
        flow_costs = [e["cost"] for e in flow_cost(original, **flow)]
        rows.append(score_variant(original, raw_costs, decoder, "shape_raw"))
        rows.append(score_variant(original, flow_costs, decoder, "shape_flow"))

        for seed in range(3):
            shuffled = shuffled_shape(original, seed)
            probabilities(model, saved["mean"], saved["std"], shuffled)
            rows.append(score_variant(shuffled, raw_costs, decoder, f"shuffle{seed}_raw"))
            rows.append(score_variant(shuffled, flow_costs, decoder, f"shuffle{seed}_flow"))
        print(alias, "complete", flush=True)

    out = pd.concat(rows, ignore_index=True)
    path = RESULTS / "family_flow_shape_factorial_h132.parquet"
    out.to_parquet(path, index=False)
    report = out.groupby(["sequence_id", "method"], sort=True).apply(
        summarize_group, include_groups=False)
    print(report.round(4).to_string(), flush=True)
    dense = out[out.n_anchors > 200].groupby(["sequence_id", "method"], sort=True).apply(
        summarize_group, include_groups=False)
    print("\nDense windows (>200 anchors):\n" + dense.round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
