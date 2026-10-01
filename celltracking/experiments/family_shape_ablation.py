"""Does per-frame mask shape help direct endpoint descendant-set inference?

Train the same pair-affinity network on location-only features, compare it with
the saved full-feature network using one frozen decoder configuration. Also
permute mask shapes among targets within each observed frame as a negative
control. Temporal GT IDs are used only to attach single-frame oracle shape and
to score results, never as model features.

Run from celltracking/: python experiments/family_shape_ablation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.family import FEATURE_NAMES, family_edges  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import (ALIASES, Affinity, BENCH, RESULTS, ROOT,
                                    examples, summarize)  # noqa: E402

LOCATION_COLUMNS = ("distance_over_spacing", "spacing_ratio", "relative_radial_distance")
LOCATION_IDX = [FEATURE_NAMES.index(name) for name in LOCATION_COLUMNS]
K_NEIGHBORS = 12


class LocationAffinity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(LOCATION_IDX), 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_location(examples_train: list[dict]):
    x = np.concatenate([e["features"][e["edge_ok"]][:, LOCATION_IDX]
                        for e in examples_train])
    y = np.concatenate([e["edge_y"][e["edge_ok"]]
                        for e in examples_train]).astype(np.float32)
    mean, std = x.mean(0), x.std(0).clip(1e-4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = LocationAffinity().to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy((x - mean) / std).to(dev)
    yt = torch.from_numpy(y).to(dev)
    for epoch in range(20):
        order = torch.randperm(len(y), device=dev)
        loss_total = 0.0
        for idx in order.split(8192):
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_total += float(loss.detach()) * len(idx)
        if epoch in (0, 19):
            print(f"location epoch {epoch}: BCE={loss_total / len(y):.4f}", flush=True)
    model.eval()
    return model, mean, std


@torch.no_grad()
def predict_edges(model, mean, std, items: list[dict], feature_idx=None):
    dev = next(model.parameters()).device
    for e in items:
        x = e["features"] if feature_idx is None else e["features"][:, feature_idx]
        normalized = torch.from_numpy((x - mean) / std).to(dev)
        e["prob"] = torch.sigmoid(model(normalized)).cpu().numpy()


def shuffled_shape(items: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    altered = []
    for e in items:
        target = e["target"].copy()
        shape_cols = ["axis_y", "axis_x", "major_sd", "minor_sd"]
        target.loc[:, shape_cols] = target[shape_cols].to_numpy()[rng.permutation(len(target))]
        edges, features = family_edges(target, K_NEIGHBORS)
        assert np.array_equal(edges, e["edges"])
        altered.append({**e, "target": target, "features": features})
    return altered


def edge_ap(items: list[dict]) -> float:
    """Micro average precision over valid target-target edges."""
    pred = np.concatenate([e["prob"][e["edge_ok"]] for e in items])
    truth = np.concatenate([e["edge_y"][e["edge_ok"]] for e in items]).astype(bool)
    order = np.argsort(-pred, kind="stable")
    ranked = truth[order]
    precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    return float(precision[ranked].mean())


def score_rows(items: list[dict], method: str, cfg: dict) -> pd.DataFrame:
    d = summarize(items, **cfg)
    d["method"] = method
    return d


def main():
    all_windows = Windows(BENCH, ROOT / "data").windows
    train_ex = [e for alias in ALIASES[:3] for e in examples(alias, all_windows, compensate=False)]
    location, lmean, lstd = train_location(train_ex)
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    full = Affinity()
    full.load_state_dict(saved["model"])
    full.eval()
    config = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())

    rows = []
    for alias in ("M3", "M4"):
        original = examples(alias, all_windows, compensate=False)
        predict_edges(full, saved["mean"], saved["std"], original)
        full_ap = edge_ap(original)
        rows.append(score_rows(original, "full_shape", config).assign(edge_ap=full_ap))

        limited = [{**e} for e in original]
        predict_edges(location, lmean, lstd, limited, LOCATION_IDX)
        loc_ap = edge_ap(limited)
        rows.append(score_rows(limited, "location_only", config).assign(edge_ap=loc_ap))

        for seed in range(3):
            permuted = shuffled_shape(original, seed)
            predict_edges(full, saved["mean"], saved["std"], permuted)
            perm_ap = edge_ap(permuted)
            rows.append(score_rows(permuted, f"shape_shuffle_{seed}", config)
                        .assign(edge_ap=perm_ap))
        print(alias, "finished", flush=True)

    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(RESULTS / "family_shape_ablation_h132.parquet", index=False)
    result["acc"] = result.n_correct / result.n_scored
    result["set_f1"] = result.set_f1_sum / result.count_n_anchors
    print(result.groupby(["sequence_id", "method"])[["acc", "set_f1", "edge_ap"]]
          .mean().round(3).to_string())
    torch.save({"model": location.cpu().state_dict(), "mean": lmean, "std": lstd,
                "features": LOCATION_COLUMNS, "train_aliases": ALIASES[:3],
                "epochs": 20, "seed": 0}, RESULTS / "family_location_only_h132.pt")


if __name__ == "__main__":
    main()
