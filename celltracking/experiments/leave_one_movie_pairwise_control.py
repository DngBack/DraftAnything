"""Movie-level robustness: direct pairwise classifier versus family center.

The pairwise design and hyperparameters were selected after inspecting M3/M4,
so this is an OIAM sensitivity audit, not a blind benchmark. Each fold trains
both models on four movies and tests the fifth across H=33/66/132. Direct
nearest association is used for both methods; no family affinity or decoder.

Run from celltracking/: python experiments/leave_one_movie_pairwise_control.py
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import ALIASES, BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402
from experiments.pairwise_geometry_baseline import (HORIZONS, pair_assignment,
                                                    score, train_classifier)  # noqa: E402


def run_fold(held, cached):
    train = [e for alias in ALIASES if alias != held for h in HORIZONS
             for e in cached[alias][h]]
    pair_model = train_classifier(train)
    center_model = fit_multi(train)
    rows = []
    for horizon in HORIZONS:
        for e in cached[held][horizon]:
            assignments = {
                "nearest_anchor": e["cost"].argmin(axis=1),
                "family_center": multi_cost(e, center_model).argmin(axis=1),
                "pairwise_geometry": pair_assignment(e, pair_model),
            }
            for name, assignment in assignments.items():
                s = score(e, assignment)
                rows.append({"held_out_movie": held, "sequence_id": held,
                             "window_id": e["window"]["window_id"],
                             "horizon_frames": horizon, "n_anchors": len(e["anchor"]),
                             "method": name, "n_scored": s["n_scored"],
                             "n_correct": s["n_correct"], "set_f1_sum": s["set_f1_sum"],
                             "count_n_anchors": s["count_n_anchors"]})
        print(held, horizon, "complete", flush=True)
    del pair_model, center_model, train
    gc.collect()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", choices=ALIASES, default=list(ALIASES))
    args = ap.parse_args()
    windows = Windows(BENCH, ROOT / "data").windows
    cached = {alias: {h: examples(alias, windows, compensate=False, horizon=h)
                      for h in HORIZONS} for alias in ALIASES}
    out = pd.concat([run_fold(alias, cached) for alias in args.folds], ignore_index=True)
    suffix = "all" if tuple(args.folds) == ALIASES else "_".join(args.folds)
    path = RESULTS / f"leave_one_movie_pairwise_control_{suffix}.parquet"
    out.to_parquet(path, index=False)
    for label, data in (("all", out), ("dense", out[out.n_anchors > 200])):
        acc = (data.n_correct / data.n_scored).groupby([data.horizon_frames, data.method]).mean()
        f1 = (data.set_f1_sum / data.count_n_anchors).groupby([data.horizon_frames, data.method]).mean()
        print(label, pd.DataFrame({"acc": acc, "f1": f1}).round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
