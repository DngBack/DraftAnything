"""Combine anonymous endpoint flow with the saved direct-family decoder.

Flow is fitted to observed point clouds only; the saved same-family network
uses single-frame mask shape. Evaluate with the decoder fixed on M3.

Run from celltracking/: python experiments/family_flow_joint_pilot.py
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import Affinity, BENCH, RESULTS, ROOT, examples, metrics, probabilities, summarize  # noqa: E402
from experiments.structure_flow_pilot import anonymous_flow, median_spacing  # noqa: E402


def flow_cost(items: list[dict], bandwidth: float, strength: float) -> list[dict]:
    out = []
    for e in items:
        anchor = e["anchor"][["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
        target = e["target"][["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
        mapped = anonymous_flow(anchor, target, bandwidth=bandwidth, strength=strength,
                                initialize="raw")
        spacing = median_spacing(anchor)
        cost = ((mapped[:, None, 1:] - anchor[None, :, 1:]) ** 2).sum(-1) / max(spacing ** 2, 1e-6)
        out.append({**e, "cost": cost.astype(np.float32)})
    return out


def main():
    config = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    model = Affinity()
    model.load_state_dict(saved["model"])
    model.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    results = []
    val = examples("M3", windows, compensate=False)
    probabilities(model, saved["mean"], saved["std"], val)
    base_val = summarize(val, **config).assign(method="family_raw")
    results.append(base_val)
    base_acc, base_f1 = metrics(base_val)
    print("M3 family_raw", tuple(round(x, 3) for x in (base_acc, base_f1)), flush=True)
    candidates = []
    for bandwidth, strength in ((2, 0.5), (4, 0.5)):
        joint = summarize(flow_cost(val, bandwidth, strength), **config)
        name = f"family_flow_h{bandwidth}_a{strength:g}"
        acc, f1 = metrics(joint)
        results.append(joint.assign(method=name))
        candidates.append((bandwidth, strength, acc, f1))
        print("M3", name, tuple(round(x, 3) for x in (acc, f1)), flush=True)
    # Choose on M3 only; require simultaneous improvement on both metrics.
    eligible = [c for c in candidates if c[2] > base_acc and c[3] > base_f1]
    selected = max(eligible, key=lambda c: min(c[2] - base_acc, c[3] - base_f1)) if eligible else None
    print("M3 selected", selected, flush=True)

    test = examples("M4", windows, compensate=False)
    probabilities(model, saved["mean"], saved["std"], test)
    base_test = summarize(test, **config).assign(method="family_raw")
    results.append(base_test)
    print("M4 family_raw", tuple(round(x, 3) for x in metrics(base_test)), flush=True)
    if selected is not None:
        bandwidth, strength = selected[:2]
        joint = summarize(flow_cost(test, bandwidth, strength), **config)
        name = f"family_flow_h{bandwidth}_a{strength:g}"
        results.append(joint.assign(method=name))
        print("M4", name, tuple(round(x, 3) for x in metrics(joint)), flush=True)
    data = pd.concat(results, ignore_index=True)
    data.to_parquet(RESULTS / "family_flow_joint_h132.parquet", index=False)
    (RESULTS / "family_flow_joint_h132_config.json").write_text(json.dumps({
        "base_config": config,
        "selected_flow": None if selected is None else {"bandwidth": selected[0],
                                                       "strength": selected[1]},
        "selection": "M3 only; simultaneous macro accuracy and set-F1 improvement",
        "M4_status": "descriptive development movie, previously viewed in other pilots",
    }, indent=2))
    print("saved", RESULTS / "family_flow_joint_h132.parquet")


if __name__ == "__main__":
    main()
