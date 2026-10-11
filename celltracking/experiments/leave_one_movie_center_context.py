"""Leave-one-movie-out control for family-center information sources.

The affinity and all center regressors are retrained in each fold. The decoder
and regressor settings come from earlier development experiments; this is a
within-OIAM robustness audit, not an independent benchmark. GT ancestry is
used for training labels and evaluation, never for held-out input features.

Run from celltracking/: python experiments/leave_one_movie_center_context.py
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_center_information_ablation import VARIANTS, cost, fit  # noqa: E402
from experiments.family_set import (ALIASES, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize, train)  # noqa: E402


def run_fold(held: str, cached: dict, decoder: dict) -> pd.DataFrame:
    train_ex = [e for alias in ALIASES if alias != held for e in cached[alias]]
    affinity, mean, std = train(train_ex)
    regressors = {name: fit(train_ex, indices, origin)
                  for name, (indices, origin) in VARIANTS.items()}
    test = [{**e} for e in cached[held]]
    probabilities(affinity, mean, std, test)
    rows = []
    for name in ("family_raw", *VARIANTS):
        data = test if name == "family_raw" else [
            {**e, "cost": cost(e, *VARIANTS[name], regressors[name])} for e in test]
        part = summarize(data, **decoder).assign(method=name)
        part["n_anchors"] = [len(e["anchor"]) for e in test]
        part["held_out_movie"] = held
        rows.append(part)
        acc = (part.n_correct / part.n_scored).mean()
        f1 = (part.set_f1_sum / part.count_n_anchors).mean()
        dense = part.n_anchors > 200
        dense_acc = (part.loc[dense, "n_correct"] / part.loc[dense, "n_scored"]).mean()
        dense_f1 = (part.loc[dense, "set_f1_sum"] / part.loc[dense, "count_n_anchors"]).mean()
        print(held, name, "all", round(acc, 4), round(f1, 4),
              "dense", round(dense_acc, 4), round(dense_f1, 4), flush=True)
    del affinity, regressors, train_ex, test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", choices=ALIASES, default=list(ALIASES))
    args = ap.parse_args()
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    windows = Windows(BENCH, ROOT / "data").windows
    cached = {alias: examples(alias, windows, compensate=False) for alias in ALIASES}
    rows = [run_fold(alias, cached, decoder) for alias in args.folds]
    result = pd.concat(rows, ignore_index=True)
    suffix = "all" if tuple(args.folds) == ALIASES else "_".join(args.folds)
    path = RESULTS / f"leave_one_movie_center_context_h132_{suffix}.parquet"
    result.to_parquet(path, index=False)
    print("saved", path, flush=True)
    for label, data in (("all", result), ("dense", result[result.n_anchors > 200])):
        acc = (data.n_correct / data.n_scored).groupby(data.method).mean()
        f1 = (data.set_f1_sum / data.count_n_anchors).groupby(data.method).mean()
        print(label, pd.DataFrame({"acc": acc, "f1": f1}).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
