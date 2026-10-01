"""Endpoint family assignment with an observable anchor-neighbourhood prior.

The saved family affinity predicts which target pairs share an anchor. A second
message-passing term encourages adjacent target *families* to map to adjacent
anchors, motivated by the separate GT-only neighbourhood survival diagnostic.
Only observed endpoint centers and per-frame oracle mask shape enter inference.

Run from celltracking/: python experiments/family_topology_pilot.py
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import Affinity, BENCH, RESULTS, ROOT, examples, metrics, probabilities, summarize  # noqa: E402


def neighbour_graph(pos: np.ndarray, k: int) -> csr_matrix:
    n = len(pos)
    if n < 2:
        return csr_matrix((n, n))
    _, near = cKDTree(pos).query(pos, k=min(k + 1, n))
    pairs = {(min(i, int(j)), max(i, int(j))) for i, row in enumerate(near)
             for j in np.atleast_1d(row)[1:] if i != j}
    if not pairs:
        return csr_matrix((n, n))
    a, b = np.asarray(sorted(pairs)).T
    graph = csr_matrix((np.ones(2 * len(a)),
                        (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    return diags(1 / np.maximum(degree, 1)) @ graph


def weighted_target_graph(n: int, edges: np.ndarray, weights: np.ndarray) -> csr_matrix:
    keep = weights > 0
    if not np.any(keep):
        return csr_matrix((n, n))
    a, b = edges[keep].T
    value = weights[keep]
    graph = csr_matrix((np.r_[value, value], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    return diags(1 / np.maximum(degree, 1e-6)) @ graph


def topology_assignment(e: dict, cfg: dict, gamma: float, anchor_k: int = 6,
                        iterations: int = 8) -> np.ndarray:
    cost = e["cost"]
    n_target, n_anchor = cost.shape
    edges, prob = e["edges"], e["prob"]
    threshold = cfg["threshold"]
    same_weight = np.clip((prob - threshold) / (1 - threshold), 0, 1)
    same_graph = weighted_target_graph(n_target, edges, same_weight)
    # An uncertain edge gets proportionally less influence than a confident
    # cross-family edge. A predicted same-family edge gets none.
    different_weight = np.clip((threshold - prob) / threshold, 0, 1)
    different_graph = weighted_target_graph(n_target, edges, different_weight)
    anchor_graph = neighbour_graph(e["anchor"][["center_y", "center_x"]].to_numpy(), anchor_k)

    unary = -cost / cfg["temperature"]
    q = softmax(unary, axis=1)
    for _ in range(iterations):
        messages = cfg["beta"] * (same_graph @ q)
        if gamma:
            messages += gamma * (different_graph @ q @ anchor_graph.T)
        q = softmax(unary + messages, axis=1)

    capacity = cfg["capacity"]
    if n_target > capacity * n_anchor:
        return q.argmax(axis=1)
    assignment_cost = -np.log(np.maximum(q, 1e-12))
    slots = np.concatenate([assignment_cost + j * cfg["extra_child_penalty"]
                            for j in range(capacity)], axis=1)
    target_rows, anchor_slots = linear_sum_assignment(slots)
    result = np.empty(n_target, dtype=np.int32)
    result[target_rows] = anchor_slots % n_anchor
    return result


def score_examples(items: list[dict], cfg: dict, gamma: float) -> pd.DataFrame:
    from cellanc.evaluate import score_window
    empty_sib = pd.DataFrame(columns=["det_1", "det_2"])
    rows = []
    for e in items:
        assignment = topology_assignment(e, cfg, gamma)
        aid = e["anchor"].local_detection_id.to_numpy(dtype=int)
        tid = e["target"].local_detection_id.to_numpy(dtype=int)
        pred = dict(zip(tid, aid[assignment]))
        s = score_window(pred, set(), e["targets"], empty_sib)
        rows.append({"sequence_id": e["window"]["sequence_id"],
                     "window_id": e["window"]["window_id"], "gamma": gamma,
                     "n_scored": s["n_scored"], "n_correct": s["n_correct"],
                     "set_f1_sum": s["set_f1_sum"], "count_n_anchors": s["count_n_anchors"]})
    return pd.DataFrame(rows)


def main():
    cfg = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    model = Affinity()
    model.load_state_dict(saved["model"])
    model.eval()
    windows = Windows(BENCH, ROOT / "data").windows

    val = examples("M3", windows, compensate=False)
    probabilities(model, saved["mean"], saved["std"], val)
    baseline = summarize(val, **cfg)
    baseline_acc, baseline_f1 = metrics(baseline)
    print("M3 existing family decoder", round(baseline_acc, 3), round(baseline_f1, 3), flush=True)
    results = [baseline.assign(method="family_raw")]
    val_scores = []
    for gamma in (0, 0.5, 1, 2, 4, 8):
        d = score_examples(val, cfg, gamma)
        acc, f1 = metrics(d)
        print("M3 gamma", gamma, round(acc, 3), round(f1, 3), flush=True)
        results.append(d.assign(method=f"topology_g{gamma:g}"))
        val_scores.append((gamma, acc, f1))
    # Selection uses validation only. Require both headline metrics to improve
    # over the gamma=0 decoder; otherwise the selected model remains gamma=0.
    eligible = [(g, acc, f1) for g, acc, f1 in val_scores
                if g > 0 and acc > baseline_acc and f1 > baseline_f1]
    selected = max(eligible, key=lambda x: min(x[1] - baseline_acc,
                                               x[2] - baseline_f1))[0] if eligible else 0.0
    print("M3 selected gamma", selected, flush=True)

    test = examples("M4", windows, compensate=False)
    probabilities(model, saved["mean"], saved["std"], test)
    results.append(summarize(test, **cfg).assign(method="family_raw"))
    results.append(score_examples(test, cfg, selected).assign(method=f"topology_g{selected:g}"))
    all_result = pd.concat(results, ignore_index=True)
    all_result.to_parquet(RESULTS / "family_topology_h132.parquet", index=False)
    (RESULTS / "family_topology_h132_config.json").write_text(json.dumps(
        {"base_config": cfg, "selected_gamma": selected, "anchor_k": 6,
         "selection": "M3 only; both macro accuracy and set F1 must improve"}, indent=2))
    for alias in ("M3", "M4"):
        for method, d in all_result[all_result.sequence_id == alias].groupby("method"):
            print(alias, method, tuple(round(v, 3) for v in metrics(d)), flush=True)


if __name__ == "__main__":
    main()
