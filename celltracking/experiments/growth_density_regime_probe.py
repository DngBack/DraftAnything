"""Post-hoc three-regime rule on already viewed development collections.

This is an analysis of saved scores, not a prospective test. If target/source
count ratio is <1.2 choose nearest; if >=8 (three doubling generations)
choose family center; otherwise choose the existing density hybrid. These
thresholds were proposed after examining recA, so no confirmatory inference
is possible from these collections.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402


def main():
    old = pd.read_parquet(RESULTS / "hybrid_rank_probe.parquet")
    old["growth_ratio"] = old.n_targets / old.n_anchors
    rec = pd.read_parquet(RESULTS / "reca_microcolony_external_test.parquet")
    rec = rec.rename(columns={"colony": "sequence", "gap_minutes": "horizon_min"})
    rec["dataset"] = "recA"
    rec["window_id"] = rec.sequence + "_" + rec.window_id
    cols = ["dataset", "sequence", "window_id", "horizon_min", "n_anchors",
            "n_targets", "growth_ratio", "method", "accuracy", "set_f1"]
    rows = pd.concat([old[cols], rec[cols]], ignore_index=True)
    keys = ["dataset", "sequence", "window_id", "horizon_min"]
    decision = rows.drop_duplicates(keys).copy()
    decision["selected_method"] = np.select(
        [decision.growth_ratio < 1.2, decision.growth_ratio >= 8],
        ["nearest_anchor", "family_center"], default="hybrid_density")
    selected = rows.merge(decision[keys + ["selected_method"]],
                          on=keys, validate="many_to_one")
    selected = selected[selected.method == selected.selected_method]
    assert len(selected) == len(decision)
    path = RESULTS / "growth_density_regime_probe.parquet"
    selected.to_parquet(path, index=False)
    print("macro window (development only)")
    print(selected.groupby(["dataset", "horizon_min"])[["accuracy", "set_f1"]]
          .mean().round(4).to_string())
    print("recA equal-colony macro")
    print(selected[selected.dataset == "recA"].groupby(
        ["horizon_min", "sequence"])[["accuracy", "set_f1"]].mean()
          .groupby(level=0).mean().round(4).to_string())
    print("saved", path)


if __name__ == "__main__":
    main()
