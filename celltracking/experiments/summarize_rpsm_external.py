"""Post-test summary of the frozen rpsM evaluation by colony and gap."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402


def main():
    data = pd.read_parquet(RESULTS / "rpsm_radial_external_test.parquet")
    keys = ["gap_frames", "gap_minutes", "colony", "method"]
    colony = data.groupby(keys, as_index=False).agg(
        macro_accuracy=("accuracy", "mean"), macro_set_f1=("set_f1", "mean"),
        windows=("window_id", "count"), scored=("n_scored", "sum"),
        excluded=("n_excluded", "sum"))
    weighted = data.groupby(keys).apply(
        lambda x: np.average(x.accuracy, weights=x.n_scored),
        include_groups=False).rename("target_weighted_accuracy").reset_index()
    colony = colony.merge(weighted, on=keys, validate="one_to_one")
    colony.to_csv(RESULTS / "rpsm_radial_colony_summary.csv", index=False)
    aggregate = colony.groupby(["gap_frames", "gap_minutes", "method"],
                               as_index=False).agg(
        colonies=("colony", "nunique"),
        equal_colony_macro_accuracy=("macro_accuracy", "mean"),
        equal_colony_macro_set_f1=("macro_set_f1", "mean"),
        equal_colony_target_weighted_accuracy=("target_weighted_accuracy", "mean"),
        windows=("windows", "sum"), scored=("scored", "sum"),
        excluded=("excluded", "sum"))
    aggregate.to_csv(RESULTS / "rpsm_radial_aggregate_summary.csv", index=False)
    print(aggregate.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
