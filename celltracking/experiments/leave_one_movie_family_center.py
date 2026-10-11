"""Movie-level robustness probe for supervised family-center registration.

For each held-out OIAM movie, retrain both same-family affinity and family-center
regressors on the other four movies. Decoder and regressor hyperparameters are
frozen from earlier M3 pilot, so this is a robustness audit, not a fresh
independent test. No held-out lineage enters training or inference features.

Run from celltracking/: python experiments/leave_one_movie_family_center.py
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import (ALIASES, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize, train)  # noqa: E402
from experiments.learned_family_center_probe import reg_cost, regression_data  # noqa: E402


def metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    return ((df.n_correct / df.n_scored).mean(),
            (df.set_f1_sum / df.count_n_anchors).mean(),
            df.n_correct.sum() / df.n_scored.sum())


def run_fold(held: str, cached: dict, decoder: dict) -> pd.DataFrame:
    train_ex = [e for alias in ALIASES if alias != held for e in cached[alias]]
    model, mean, std = train(train_ex)
    x, y = regression_data(train_ex)
    regressors = [HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05,
                  max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1,
                  random_state=0).fit(x, y[:, j]) for j in range(2)]
    test = [{**e} for e in cached[held]]
    probabilities(model, mean, std, test)
    baseline = summarize(test, **decoder).assign(method="family_raw")
    transformed = [{**e, "cost": reg_cost(e, regressors)} for e in test]
    proposed = summarize(transformed, **decoder).assign(method="learned_center")
    for part in (baseline, proposed):
        part["n_anchors"] = [len(e["anchor"]) for e in test]
        part["held_out_movie"] = held
        part["train_movies"] = ",".join(a for a in ALIASES if a != held)
    print(held, "raw", tuple(round(float(v), 4) for v in metrics(baseline)),
          "learned", tuple(round(float(v), 4) for v in metrics(proposed)), flush=True)
    del model, regressors, train_ex, test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat([baseline, proposed], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", choices=ALIASES, default=list(ALIASES))
    args = ap.parse_args()
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    windows = Windows(BENCH, ROOT / "data").windows
    cached = {alias: examples(alias, windows, compensate=False) for alias in ALIASES}
    rows = []
    for held in args.folds:
        rows.append(run_fold(held, cached, decoder))
    out = pd.concat(rows, ignore_index=True)
    suffix = "all" if tuple(args.folds) == ALIASES else "_".join(args.folds)
    path = RESULTS / f"leave_one_movie_family_center_h132_{suffix}.parquet"
    out.to_parquet(path, index=False)
    print("saved", path, flush=True)
    for name, subset in (("all", out), ("dense", out[out.n_anchors > 200])):
        print(name)
        for method, df in subset.groupby("method"):
            print(method, tuple(round(float(v), 4) for v in metrics(df)), flush=True)


if __name__ == "__main__":
    main()
