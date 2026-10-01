"""Pair-family refinement on raw endpoint coordinates, selecting for dense as well as easy windows.

The OIAM colony-expansion correction can harm crowded late windows. This
experiment holds the learned family affinity fixed and tunes only its decoder
on M3. M4 is read after the M3 choice is fixed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples, metrics,
                                    probabilities, summarize)  # noqa: E402
from experiments.family_set_capacity import established_baselines  # noqa: E402


def pooled_acc(df: pd.DataFrame) -> float:
    return float(df.n_correct.sum() / df.n_scored.sum())


def main():
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    model = Affinity()
    model.load_state_dict(saved["model"])
    model.eval()
    all_windows = Windows(BENCH, ROOT / "data").windows
    val_ex = examples("M3", all_windows, compensate=False)
    probabilities(model, saved["mean"], saved["std"], val_ex)
    reference = established_baselines("M3", all_windows)
    reference = reference[reference.method == "chained_lap_c4"]
    base_acc, base_f1 = metrics(reference)
    base_pool = pooled_acc(reference)
    trials = []
    for beta in (0, 2, 4, 8):
        for threshold in (0.3, 0.5):
            for temperature in (1, 2, 4):
                for penalty in (0, 0.5, 1):
                    config = {"beta": beta, "threshold": threshold,
                              "temperature": temperature, "capacity": 4,
                              "extra_child_penalty": penalty}
                    df = summarize(val_ex, **config)
                    acc, f1 = metrics(df)
                    pooled = pooled_acc(df)
                    robust_ratio = min(acc / base_acc, f1 / base_f1, pooled / base_pool)
                    trials.append({**config, "acc": acc, "set_f1": f1,
                                   "pooled_acc": pooled, "robust_ratio": robust_ratio})
    rank = pd.DataFrame(trials).sort_values(["robust_ratio", "set_f1"], ascending=False)
    best = rank.iloc[0]
    config = {k: int(best[k]) if k == "capacity" else float(best[k]) for k in
              ("beta", "threshold", "temperature", "capacity", "extra_child_penalty")}
    print("M3 LAP c4 reference:", tuple(round(x, 3) for x in (base_acc, base_f1, base_pool)))
    print("M3 selected:", config, "metrics", tuple(round(float(best[k]), 3) for k in
          ("acc", "set_f1", "pooled_acc", "robust_ratio")), flush=True)

    test_ex = examples("M4", all_windows, compensate=False)
    probabilities(model, saved["mean"], saved["std"], test_ex)
    rows = []
    for alias, ex in (("M3", val_ex), ("M4", test_ex)):
        base = established_baselines(alias, all_windows)
        chosen = summarize(ex, **config).assign(method="family_set_raw")
        rows.extend((base, chosen))
        for name, df in pd.concat([base, chosen]).groupby("method"):
            print(alias, name, "macro_acc/F1/pooled_acc",
                  tuple(round(x, 3) for x in (*metrics(df), pooled_acc(df))), flush=True)
    pd.concat(rows).to_parquet(RESULTS / "family_set_h132_raw_windows.parquet", index=False)
    rank.to_csv(RESULTS / "family_set_h132_raw_val_grid.csv", index=False)
    (RESULTS / "family_set_h132_raw_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
