"""First no-new-data probes for sparse ancestry on One-in-a-Million.

Topology is a GT-only diagnostic: it asks whether the neighbourhood of an
anchor survives when each anchor is represented by its descendants' centroid.
Flow is an inference experiment: it estimates an anonymous displacement field
from the two *observed* point clouds, then applies the existing c4 LAP decoder.
Ground truth is read only to score the flow experiment.

Run from celltracking/:
  python experiments/structure_flow_pilot.py topology
  python experiments/structure_flow_pilot.py flow --seqs M3
  python experiments/structure_flow_pilot.py flow --seqs M4 --config 4:0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import _lap, expansion_compensate  # noqa: E402
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from cellanc.transport import log_plan_from_cost  # noqa: E402

ROOT = Path(__file__).parents[1]
BENCH = ROOT / "outputs/benchmarks/v1"
RESULTS = ROOT / "outputs/results"
WINDOWS = Windows(BENCH, ROOT / "data")
EMPTY_SIBLINGS = pd.DataFrame(columns=["det_1", "det_2"])


def endpoint_windows(alias: str, horizons: tuple[int, ...]) -> list[dict]:
    return [w for w in WINDOWS.windows if w["sequence_id"] == alias
            and w["horizon_frames"] in horizons and w["schedule_id"] == "ends"
            and f"A_h{w['horizon_frames']}" in w["experiments"]]


def target_labels(alias: str) -> dict[str, pd.DataFrame]:
    path = BENCH / "labels/targets" / f"{alias}.parquet"
    return dict(list(pd.read_parquet(path).groupby("window_id")))


def neighbour_edges(pos: np.ndarray, k: int) -> set[tuple[int, int]]:
    near = cKDTree(pos).query(pos, k=min(k + 1, len(pos)))[1]
    return {tuple(sorted((i, int(j)))) for i, row in enumerate(near)
            for j in np.atleast_1d(row)[1:] if i != j}


def topology_window(win: dict, targets: pd.DataFrame, k: int, seed: int) -> dict | None:
    w = win["window"]
    src = win["detections"][w["anchor_frame"]]
    dst = win["detections"][w["target_frame"]]
    valid = targets[targets.in_foi & (targets.target_status == "valid_anchor")]
    by_target = dict(zip(dst[:, 0].astype(int), dst[:, 1:]))
    centroids = valid.groupby("ancestor_detection_id").target_detection_id.apply(
        lambda ids: np.mean([by_target[int(i)] for i in ids], axis=0))
    anchors = src[:, 0].astype(int)
    keep = np.array([i in centroids.index for i in anchors])
    if keep.sum() < max(8, k + 2):
        return None
    old = src[keep, 1:]
    new = np.stack([centroids[i] for i in anchors[keep]])
    e_old = neighbour_edges(old, k)
    e_new = neighbour_edges(new, k)
    recall = len(e_old & e_new) / len(e_old)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(20):
        perm = rng.permutation(len(new))
        e_rand = {tuple(sorted((int(perm[i]), int(perm[j])))) for i, j in e_new}
        null.append(len(e_old & e_rand) / len(e_old))
    spacing = np.median(cKDTree(old).query(old, k=2)[0][:, 1])
    displacement = np.linalg.norm(new - old, axis=1) / max(spacing, 1e-6)
    return {"sequence_id": w["sequence_id"], "window_id": w["window_id"],
            "horizon_frames": w["horizon_frames"], "anchor_frame": w["anchor_frame"],
            "n_anchors": len(src), "n_survivors": len(old), "edge_recall": recall,
            "null_edge_recall": float(np.mean(null)),
            "median_displacement_over_spacing": float(np.median(displacement))}


def run_topology(aliases: list[str], horizons: tuple[int, ...], k: int) -> None:
    rows = []
    for alias in aliases:
        labels = target_labels(alias)
        for w in endpoint_windows(alias, horizons):
            result = topology_window(WINDOWS.load(w), labels[w["window_id"]], k, seed=0)
            if result is not None:
                rows.append(result)
        print(alias, "finished", flush=True)
    out = RESULTS / f"topology_k{k}.parquet"
    data = pd.DataFrame(rows)
    data.to_parquet(out, index=False)
    print(data.groupby(["sequence_id", "horizon_frames"])[
        ["edge_recall", "null_edge_recall", "median_displacement_over_spacing"]]
        .agg(["mean", "count"]).round(3).to_string())
    print("saved", out)


def median_spacing(src: np.ndarray) -> float:
    return float(np.median(cKDTree(src[:, 1:]).query(src[:, 1:], k=2)[0][:, 1]))


def anonymous_flow(src: np.ndarray, dst: np.ndarray, *, bandwidth: float,
                   strength: float, initialize: str = "rc",
                   eps: float = 1.0, tau: float = 3.0) -> np.ndarray:
    """Map dst to src coordinates using smoothed, anonymous transport residuals.

    The Sinkhorn plan is used only to estimate a spatial field. Final identities
    come from the same c4 LAP decoder as the baseline. No GT IDs/masks enter.
    """
    adjusted = expansion_compensate(src, dst) if initialize == "rc" else dst.astype(float).copy()
    if len(src) < 2 or len(dst) < 2:
        return adjusted
    spacing = median_spacing(src)
    cost = ((src[:, None, 1:] - adjusted[None, :, 1:]) ** 2).sum(-1) / max(spacing ** 2, 1e-6)
    lp = log_plan_from_cost(cost, np.ones(len(src)), np.ones(len(dst)),
                            eps, tau, iters=120, tol=1e-3)
    parent_prob = np.exp(lp - logsumexp(lp, axis=0, keepdims=True))
    barycentre = parent_prob.T @ src[:, 1:]
    residual = barycentre - adjusted[:, 1:]
    xy = adjusted[:, 1:]
    tree = cKDTree(xy)
    radius = 3 * bandwidth * spacing
    neighbours = tree.query_ball_point(xy, r=radius)
    smoothed = np.empty_like(residual)
    for i, near in enumerate(neighbours):
        d2 = ((xy[near] - xy[i]) ** 2).sum(axis=1)
        weights = np.exp(-d2 / (2 * (bandwidth * spacing) ** 2))
        smoothed[i] = weights @ residual[near] / weights.sum()
    out = adjusted.copy()
    out[:, 1:] += strength * smoothed
    return out


def prediction(src: np.ndarray, dst: np.ndarray, mode: str,
               bandwidth: float = 4, strength: float = 0.5) -> dict[int, int]:
    if mode == "raw":
        mapped = dst
    elif mode == "rc":
        mapped = expansion_compensate(src, dst)
    else:
        mapped = anonymous_flow(src, dst, bandwidth=bandwidth, strength=strength,
                                initialize="raw" if mode == "flow_raw" else "rc")
    parent, _ = _lap(src, mapped, cap=4)
    return dict(zip(dst[:, 0].astype(int), parent.astype(int)))


def run_flow(aliases: list[str], horizon: int, configs: list[tuple[float, float]]) -> None:
    rows = []
    for alias in aliases:
        labels = target_labels(alias)
        for w in endpoint_windows(alias, (horizon,)):
            win = WINDOWS.load(w)
            src = win["detections"][w["anchor_frame"]]
            dst = win["detections"][w["target_frame"]]
            tg = labels[w["window_id"]]
            variants = [("lap_c4_raw", "raw", 0, 0), ("lap_c4_rc", "rc", 0, 0)]
            variants += [(f"flow_{init}_h{bw:g}_a{alpha:g}", f"flow_{init}", bw, alpha)
                         for init in ("raw", "rc") for bw, alpha in configs]
            for name, mode, bw, alpha in variants:
                pred = prediction(src, dst, mode, bw, alpha)
                score = score_window(pred, set(), tg, EMPTY_SIBLINGS)
                rows.append({"sequence_id": alias, "window_id": w["window_id"],
                             "anchor_frame": w["anchor_frame"], "n_anchors": len(src),
                             "method": name, "n_scored": score["n_scored"],
                             "n_correct": score["n_correct"],
                             "set_f1_sum": score["set_f1_sum"],
                             "count_n_anchors": score["count_n_anchors"]})
        print(alias, "finished", flush=True)
    out = RESULTS / f"anonymous_flow_h{horizon}_{'_'.join(aliases)}.parquet"
    data = pd.DataFrame(rows)
    data.to_parquet(out, index=False)
    data["acc"] = data.n_correct / data.n_scored
    data["set_f1"] = data.set_f1_sum / data.count_n_anchors
    summary = data.groupby(["sequence_id", "method"]).agg(
        macro_acc=("acc", "mean"), mean_set_f1=("set_f1", "mean"),
        correct=("n_correct", "sum"), scored=("n_scored", "sum"), n_windows=("acc", "size"))
    summary["pooled_acc"] = summary.correct / summary.scored
    print(summary[["macro_acc", "mean_set_f1", "pooled_acc", "n_windows"]]
          .round(3).to_string())
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["topology", "flow"])
    ap.add_argument("--seqs", nargs="+", default=None)
    ap.add_argument("--horizons", nargs="+", type=int, default=[33, 66, 132])
    ap.add_argument("--H", type=int, default=132)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--config", nargs="+", default=["2:0.5", "4:0.5", "8:0.5", "4:1"])
    args = ap.parse_args()
    if args.probe == "topology":
        run_topology(args.seqs or ["M0", "M1", "M2", "M3"], tuple(args.horizons), args.k)
    else:
        configs = [tuple(map(float, c.split(":"))) for c in args.config]
        run_flow(args.seqs or ["M3"], args.H, configs)
