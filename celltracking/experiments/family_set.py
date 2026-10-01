"""Pilot: direct descendant-set inference on existing OIAM H=132 endpoint windows.

Train a same-ancestor affinity from M0-M2 target-frame geometry. Choose the
collective assignment strength on M3, then report M4 once with that choice.
Oracle centers and per-frame mask shape are the input condition. GT identities
are used only to align per-frame shape values and to form train/evaluator labels.

  python experiments/family_set.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.family import FEATURE_NAMES, collective_assignment, endpoint_cost, family_edges  # noqa: E402
from cellanc.loader import Windows  # noqa: E402

ROOT = Path(__file__).parents[1]
BENCH = ROOT / "outputs/benchmarks/v1"
RESULTS = ROOT / "outputs/results"
ALIASES = ("M0", "M1", "M2", "M3", "M4")
SEQUENCE = {f"M{i}": f"0{i}" for i in range(5)}
HORIZON = 132
K_NEIGHBORS = 12


class Affinity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_alias(alias: str, windows: list[dict]) -> tuple[dict[int, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Join observable per-frame mask shape to relabelled detection IDs."""
    frames = {t for w in windows for t in (w["anchor_frame"], w["target_frame"])}
    det = pd.read_parquet(BENCH / "inputs/detections" / f"{alias}.parquet")
    det = det[det.frame_index.isin(frames)]
    mapping = pd.read_parquet(BENCH / "labels/gt_mapping" / f"{alias}.parquet",
                              columns=["frame_index", "local_detection_id", "gt_track_id"])
    mapping = mapping[mapping.frame_index.isin(frames)]
    shape = pd.read_parquet(ROOT / "outputs/index/one_in_a_million" / SEQUENCE[alias] / "shape.parquet")
    shape = shape[shape.frame_index.isin(frames)]
    # Temporal identity never enters family_edges or endpoint_cost. The join is solely
    # a convenience for attaching mask-derived single-frame second moments.
    det = det.merge(mapping, on=["frame_index", "local_detection_id"], validate="one_to_one")
    det = det.merge(shape, on=["frame_index", "gt_track_id"], validate="one_to_one")
    by_frame = {int(t): g.sort_values("local_detection_id").reset_index(drop=True)
                for t, g in det.groupby("frame_index")}
    tg = pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
    return by_frame, dict(list(tg.groupby("window_id")))


def windows_for(alias: str, all_windows: list[dict]) -> list[dict]:
    return [w for w in all_windows if w["sequence_id"] == alias and
            w["horizon_frames"] == HORIZON and w["schedule_id"] == "ends" and
            f"A_h{HORIZON}" in w["experiments"]]


def examples(alias: str, all_windows: list[dict], compensate: bool = True) -> list[dict]:
    windows = windows_for(alias, all_windows)
    by_frame, targets = load_alias(alias, windows)
    out = []
    for w in windows:
        anchor, target = by_frame[w["anchor_frame"]], by_frame[w["target_frame"]]
        edges, features = family_edges(target, K_NEIGHBORS)
        tg = targets[w["window_id"]]
        valid = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
        ancestry = dict(zip(valid.target_detection_id, valid.ancestor_detection_id))
        labels = np.array([ancestry.get(int(i), -1) for i in target.local_detection_id])
        edge_ok = (labels[edges[:, 0]] >= 0) & (labels[edges[:, 1]] >= 0)
        edge_y = labels[edges[:, 0]] == labels[edges[:, 1]]
        out.append({"window": w, "anchor": anchor, "target": target, "targets": tg,
                    "edges": edges, "features": features, "edge_ok": edge_ok,
                    "edge_y": edge_y, "cost": endpoint_cost(anchor, target, compensate)})
    return out


def train(train_examples: list[dict]) -> tuple[Affinity, np.ndarray, np.ndarray]:
    x = np.concatenate([e["features"][e["edge_ok"]] for e in train_examples])
    y = np.concatenate([e["edge_y"][e["edge_ok"]] for e in train_examples]).astype(np.float32)
    mean, std = x.mean(0), x.std(0).clip(1e-4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = Affinity().to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_tensor = torch.from_numpy((x - mean) / std).to(dev)
    y_tensor = torch.from_numpy(y).to(dev)
    for epoch in range(20):
        order = torch.randperm(len(y), device=dev)
        total = 0.0
        for idx in order.split(8192):
            logits = model(x_tensor[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach()) * len(idx)
        if epoch in (0, 4, 9, 19):
            print(f"edge epoch {epoch}: BCE={total / len(y):.4f}", flush=True)
    print(f"train edges={len(y)}, positive_rate={y.mean():.3f}", flush=True)
    model.eval()
    return model, mean, std


@torch.no_grad()
def probabilities(model: Affinity, mean: np.ndarray, std: np.ndarray, ex: list[dict]) -> None:
    dev = next(model.parameters()).device
    for e in ex:
        x = torch.from_numpy((e["features"] - mean) / std).to(dev)
        e["prob"] = torch.sigmoid(model(x)).cpu().numpy()


def summarize(ex: list[dict], beta: float, threshold: float, temperature: float,
              capacity: int | None = None, extra_child_penalty: float = 0.0) -> pd.DataFrame:
    rows = []
    empty_sib = pd.DataFrame(columns=["det_1", "det_2"])
    for e in ex:
        assignment = collective_assignment(e["cost"], e["edges"], e["prob"],
                                            beta, threshold, temperature, capacity=capacity,
                                            extra_child_penalty=extra_child_penalty)
        aid = e["anchor"].local_detection_id.to_numpy(dtype=int)
        tid = e["target"].local_detection_id.to_numpy(dtype=int)
        pred = dict(zip(tid, aid[assignment]))
        s = score_window(pred, set(), e["targets"], empty_sib)
        rows.append({"sequence_id": e["window"]["sequence_id"],
                     "window_id": e["window"]["window_id"],
                     "beta": beta, "threshold": threshold, "temperature": temperature,
                     "capacity": capacity, "extra_child_penalty": extra_child_penalty,
                     "n_scored": s["n_scored"], "n_correct": s["n_correct"],
                     "set_f1_sum": s["set_f1_sum"], "count_n_anchors": s["count_n_anchors"]})
    return pd.DataFrame(rows)


def metrics(df: pd.DataFrame) -> tuple[float, float]:
    acc = (df.n_correct / df.n_scored).mean()
    set_f1 = (df.set_f1_sum / df.count_n_anchors).mean()
    return float(acc), float(set_f1)


def main():
    all_windows = Windows(BENCH, ROOT / "data").windows
    train_ex = [e for alias in ALIASES[:3] for e in examples(alias, all_windows)]
    model, mean, std = train(train_ex)
    val_ex = examples("M3", all_windows)
    probabilities(model, mean, std, val_ex)
    valid_edge = np.concatenate([e["edge_y"][e["edge_ok"]] for e in val_ex])
    valid_prob = np.concatenate([e["prob"][e["edge_ok"]] for e in val_ex])
    for threshold in (0.3, 0.5, 0.7):
        selected = valid_prob >= threshold
        print(f"M3 edge threshold={threshold}: precision={valid_edge[selected].mean():.3f}, "
              f"recall={selected[valid_edge].mean():.3f}, edges={selected.sum()}", flush=True)

    baseline = summarize(val_ex, 0, 0.5, 1)
    baseline_acc, baseline_f1 = metrics(baseline)
    print(f"M3 nearest-anchor-rc: acc={baseline_acc:.3f}, F1={baseline_f1:.3f}", flush=True)
    trials = []
    for beta in (0.5, 1, 2, 4, 8, 16):
        for threshold in (0.3, 0.5, 0.7):
            for temperature in (2, 4, 8, 16):
                df = summarize(val_ex, beta, threshold, temperature)
                acc, f1 = metrics(df)
                trials.append({"beta": beta, "threshold": threshold,
                               "temperature": temperature, "acc": acc, "set_f1": f1})
    rank = pd.DataFrame(trials).sort_values(["set_f1", "acc"], ascending=False)
    best = rank.iloc[0]
    config = {"beta": float(best.beta), "threshold": float(best.threshold),
              "temperature": float(best.temperature)}
    print("M3 selected", config, f"acc={best.acc:.3f}, F1={best.set_f1:.3f}", flush=True)

    # M4 is loaded only after all hyperparameters have been fixed on M3.
    test_ex = examples("M4", all_windows)
    probabilities(model, mean, std, test_ex)
    test_base = summarize(test_ex, 0, 0.5, 1)
    test_new = summarize(test_ex, **config)
    print("M4 nearest-anchor-rc", tuple(round(x, 3) for x in metrics(test_base)), flush=True)
    print("M4 family-set", tuple(round(x, 3) for x in metrics(test_new)), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "mean": mean, "std": std,
                "features": FEATURE_NAMES, "train_aliases": ALIASES[:3]},
               RESULTS / "family_set_h132.pt")
    rank.to_csv(RESULTS / "family_set_h132_val_grid.csv", index=False)
    pd.concat([baseline.assign(method="nearest_anchor_rc"),
               summarize(val_ex, **config).assign(method="family_set"),
               test_base.assign(method="nearest_anchor_rc"),
               test_new.assign(method="family_set")]).to_parquet(
                   RESULTS / "family_set_h132_windows.parquet", index=False)
    (RESULTS / "family_set_h132_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
