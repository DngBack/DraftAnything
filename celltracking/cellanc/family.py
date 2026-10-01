"""Target-family affinity and soft collective assignment for sparse ancestry.

Features use only information available in one observed frame: cell centers and
per-instance second moments from an oracle mask. Temporal IDs are never features.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree
from scipy.special import softmax

from .baselines import expansion_compensate


FEATURE_NAMES = (
    "distance_over_spacing", "distance_over_mean_length", "axis_alignment",
    "displacement_along_axis_1", "displacement_along_axis_2",
    "log_length_ratio", "log_width_ratio", "log_aspect_ratio_difference",
    "spacing_ratio", "relative_radial_distance",
)


def family_edges(frame: pd.DataFrame, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return undirected kNN edges and their observable geometry features.

    The rows in `frame` define edge indices. Its local IDs may be used to align
    labels by the caller, but GT track IDs and ancestry are never inspected here.
    """
    n = len(frame)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    pos = frame[["center_y", "center_x"]].to_numpy(dtype=np.float64)
    dist, near = cKDTree(pos).query(pos, k=min(k + 1, n))
    spacing = np.maximum(dist[:, 1], 1e-3)
    edges = np.array(sorted({(min(i, int(j)), max(i, int(j)))
                             for i in range(n) for j in np.atleast_2d(near)[i, 1:]
                             if i != j}), dtype=np.int32).reshape(-1, 2)
    a, b = edges.T
    delta = pos[b] - pos[a]
    length = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
    unit_delta = delta / length[:, None]
    axis = frame[["axis_y", "axis_x"]].fillna(0).to_numpy(dtype=np.float64).copy()
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-6)
    major = np.maximum(frame["major_sd"].fillna(1).to_numpy(dtype=np.float64), 1e-3)
    minor = np.maximum(frame["minor_sd"].fillna(1).to_numpy(dtype=np.float64), 1e-3)
    mean_pos = pos.mean(axis=0)
    radial = np.linalg.norm(pos - mean_pos, axis=1)
    radial_scale = np.sqrt(np.mean(radial ** 2)) + 1e-6
    features = np.column_stack([
        length / np.sqrt(spacing[a] * spacing[b]),
        length / np.sqrt(major[a] * major[b]),
        np.abs(np.sum(axis[a] * axis[b], axis=1)),
        np.abs(np.sum(unit_delta * axis[a], axis=1)),
        np.abs(np.sum(unit_delta * axis[b], axis=1)),
        np.abs(np.log(major[a] / major[b])),
        np.abs(np.log(minor[a] / minor[b])),
        np.abs(np.log(major[a] / minor[a]) - np.log(major[b] / minor[b])),
        np.abs(np.log(spacing[a] / spacing[b])),
        np.abs(radial[a] - radial[b]) / radial_scale,
    ])
    return edges, features.astype(np.float32)


def endpoint_cost(anchor: pd.DataFrame, target: pd.DataFrame,
                  compensate: bool = True) -> np.ndarray:
    """Squared endpoint distance, scaled by anchor spacing."""
    src = anchor[["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
    dst = target[["local_detection_id", "center_y", "center_x"]].to_numpy(dtype=float)
    adjusted = expansion_compensate(src, dst) if compensate else dst
    spacing = np.median(cKDTree(src[:, 1:]).query(src[:, 1:], k=2)[0][:, 1])
    return (((adjusted[:, None, 1:] - src[None, :, 1:]) ** 2).sum(-1) /
            max(spacing ** 2, 1e-6)).astype(np.float32)


def collective_assignment(cost: np.ndarray, edges: np.ndarray, prob: np.ndarray,
                          beta: float, threshold: float, temperature: float,
                          iterations: int = 8, capacity: int | None = None,
                          extra_child_penalty: float = 0.0) -> np.ndarray:
    """Mean-field assignment with a learned soft same-family graph.

    For beta=0 this is exactly nearest-anchor assignment on `cost`. An edge's
    contribution is positive only above `threshold`, and its row normalization
    keeps the strength comparable across sparse and crowded frames.
    """
    if (beta == 0 or len(edges) == 0) and capacity is None:
        return cost.argmin(axis=1)
    n_target = len(cost)
    unary = -cost / temperature
    q = softmax(unary, axis=1)
    if beta != 0 and len(edges):
        weight = np.clip((prob - threshold) / max(1 - threshold, 1e-6), 0, 1)
        keep = weight > 0
        if keep.any():
            a, b = edges[keep].T
            rows = np.concatenate([a, b])
            cols = np.concatenate([b, a])
            values = np.tile(weight[keep], 2)
            graph = csr_matrix((values, (rows, cols)), shape=(n_target, n_target))
            degree = np.asarray(graph.sum(axis=1)).ravel()
            graph = diags(1 / np.maximum(degree, 1e-6)) @ graph
            for _ in range(iterations):
                q = softmax(unary + beta * (graph @ q), axis=1)
    if capacity is None or n_target > capacity * cost.shape[1]:
        return q.argmax(axis=1)
    assignment_cost = -np.log(np.maximum(q, 1e-12))
    slots = np.concatenate([assignment_cost + j * extra_child_penalty
                            for j in range(capacity)], axis=1)
    target_rows, anchor_slots = linear_sum_assignment(slots)
    result = np.empty(n_target, dtype=np.int32)
    result[target_rows] = anchor_slots % cost.shape[1]
    return result
