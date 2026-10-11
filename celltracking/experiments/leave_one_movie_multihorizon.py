"""Movie-level robustness audit of horizon-conditioned center registration.

Each fold retrains the H=132 family affinity and both pooled center regressors
on four OIAM movies, then evaluates the fifth at H=33,66,132. Hyperparameters
and design were selected after inspecting these movies in earlier pilots, so
the folds are descriptive sensitivity checks, not fresh blind tests.

Run from celltracking/: python experiments/leave_one_movie_multihorizon.py
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
from experiments.family_set import (ALIASES, BENCH, RESULTS, ROOT, examples,
                                    probabilities, summarize, train)  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402


HORIZONS = (33, 66, 132)


def score(items, decoder, name):
    part = summarize(items, **decoder).assign(method=name)
    part["n_anchors"] = [len(e["anchor"]) for e in items]
    return part


def run_fold(held, cached, decoder):
    train132 = [e for alias in ALIASES if alias != held for e in cached[alias][132]]
    trainall = [e for alias in ALIASES if alias != held for h in HORIZONS
                for e in cached[alias][h]]
    affinity, mean, std = train(train132)
    pooled = fit_multi(trainall, include_horizon=False)
    multi = fit_multi(trainall, include_horizon=True)
    rows = []
    for horizon in HORIZONS:
        test = [{**e} for e in cached[held][horizon]]
        probabilities(affinity, mean, std, test)
        variants = {
            "family_raw": test,
            "pooled_no_horizon": [{**e, "cost": multi_cost(e, pooled, False)} for e in test],
            "multi_horizon": [{**e, "cost": multi_cost(e, multi, True)} for e in test],
        }
        for name, items in variants.items():
            part = score(items, decoder, name).assign(held_out_movie=held,
                                                      horizon_frames=horizon)
            rows.append(part)
            acc = (part.n_correct / part.n_scored).mean()
            f1 = (part.set_f1_sum / part.count_n_anchors).mean()
            print(held, horizon, name, round(acc, 4), round(f1, 4), flush=True)
    del affinity, pooled, multi, train132, trainall
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
    cached = {alias: {h: examples(alias, windows, compensate=False, horizon=h)
                      for h in HORIZONS} for alias in ALIASES}
    result = pd.concat([run_fold(alias, cached, decoder) for alias in args.folds],
                       ignore_index=True)
    suffix = "all" if tuple(args.folds) == ALIASES else "_".join(args.folds)
    path = RESULTS / f"leave_one_movie_multihorizon_{suffix}.parquet"
    result.to_parquet(path, index=False)
    for label, data in (("all", result), ("dense", result[result.n_anchors > 200])):
        acc = (data.n_correct / data.n_scored).groupby([data.horizon_frames, data.method]).mean()
        f1 = (data.set_f1_sum / data.count_n_anchors).groupby([data.horizon_frames, data.method]).mean()
        print(label, pd.DataFrame({"acc": acc, "f1": f1}).round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
