"""Frozen H=132 family-center model on shorter and longer endpoint gaps.

Train family-center regressors on M0-M2 H=132 ends. Reuse the saved H=132
affinity/decoder and evaluate M3/M4 at H=33,66,132,264. This tests temporal
transfer within previously viewed OIAM movies, not independent generalization.

Run from celltracking/: python experiments/cross_horizon_center_context.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_center_information_ablation import (SOURCE, cost,
                                                            fit)  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize)  # noqa: E402


SHIFT = sorted(SOURCE + [2, 3, 8, 9])


def main():
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    affinity = Affinity()
    affinity.load_state_dict(saved["model"])
    affinity.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for e in examples(alias, windows, compensate=False)]
    models = {"source_prior": fit(train, SOURCE, "source"),
              "target_shift": fit(train, SHIFT, "source")}
    rows = []
    for alias in ("M3", "M4"):
        for horizon in (33, 66, 132, 264):
            items = examples(alias, windows, compensate=False, horizon=horizon)
            if not items:
                continue
            probabilities(affinity, saved["mean"], saved["std"], items)
            for name in ("family_raw", "source_prior", "target_shift"):
                data = items if name == "family_raw" else [
                    {**e, "cost": cost(e, SOURCE if name == "source_prior" else SHIFT,
                                      "source", models[name])} for e in items]
                part = summarize(data, **decoder).assign(method=name, horizon_frames=horizon)
                part["n_anchors"] = [len(e["anchor"]) for e in items]
                rows.append(part)
            print(alias, horizon, len(items), "complete", flush=True)
    out = pd.concat(rows, ignore_index=True)
    path = RESULTS / "cross_horizon_center_context.parquet"
    out.to_parquet(path, index=False)
    for label, data in (("all", out), ("dense", out[out.n_anchors > 200])):
        acc = (data.n_correct / data.n_scored).groupby([data.horizon_frames, data.method]).mean()
        f1 = (data.set_f1_sum / data.count_n_anchors).groupby([data.horizon_frames, data.method]).mean()
        print(label, pd.DataFrame({"acc": acc, "f1": f1}).round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
