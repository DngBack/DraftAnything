"""Summarize the frozen recA evaluation by biological replicate.

This reads saved predictions only. It does not change model or primary metric.
Run from celltracking/: python experiments/summarize_reca_external.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402


def main():
    data = pd.read_parquet(RESULTS / "reca_microcolony_external_test.parquet")
    cols = ["gap_frames", "gap_minutes", "colony", "method"]
    colony = data.groupby(cols, as_index=False).agg(
        macro_accuracy=("accuracy", "mean"),
        macro_set_f1=("set_f1", "mean"),
        n_windows=("window_id", "count"),
        n_scored=("n_scored", "sum"),
        n_excluded=("n_excluded", "sum"))
    weighted = data.groupby(cols).apply(
        lambda x: np.average(x.accuracy, weights=x.n_scored),
        include_groups=False).rename("target_weighted_accuracy").reset_index()
    colony = colony.merge(weighted, on=cols, validate="one_to_one")
    colony.to_csv(RESULTS / "reca_microcolony_colony_summary.csv", index=False)
    aggregate = colony.groupby(["gap_frames", "gap_minutes", "method"],
                               as_index=False).agg(
        colonies=("colony", "nunique"),
        equal_colony_macro_accuracy=("macro_accuracy", "mean"),
        equal_colony_macro_set_f1=("macro_set_f1", "mean"),
        equal_colony_target_weighted_accuracy=("target_weighted_accuracy", "mean"),
        windows=("n_windows", "sum"),
        scored=("n_scored", "sum"),
        excluded=("n_excluded", "sum"))
    aggregate.to_csv(RESULTS / "reca_microcolony_aggregate_summary.csv", index=False)
    print(aggregate.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
