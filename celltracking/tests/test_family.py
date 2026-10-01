"""Guards against temporal-ID leakage and checks collective assignment invariants."""

import numpy as np
import pandas as pd

from cellanc.family import collective_assignment, family_edges
from experiments.train_embed import frame_features


def _frame(ids, y):
    n = len(ids)
    return pd.DataFrame({"gt_track_id": ids, "center_y": y, "center_x": np.zeros(n),
                         "axis_y": np.ones(n), "axis_x": np.zeros(n),
                         "major_sd": np.full(n, 2.0), "minor_sd": np.ones(n)})


def test_clean_embedding_features_ignore_temporal_ids():
    current = _frame([1, 2, 3], [1.0, 11.0, 21.0])
    previous = _frame([1, 2, 3], [0.0, 10.0, 20.0])
    _, clean = frame_features(current, previous, 1)
    _, permuted = frame_features(current.assign(gt_track_id=[3, 1, 2]), previous, 1)
    np.testing.assert_allclose(clean, permuted)
    np.testing.assert_array_equal(clean[:, 7:10], np.zeros((3, 3)))
    _, diagnostic = frame_features(current, previous, 1, oracle_velocity=True)
    assert diagnostic[:, 9].sum() == 3  # legacy mode really uses GT identity


def test_family_edges_ignore_temporal_ids():
    frame = _frame([1, 2, 3, 4], [0.0, 2.0, 4.0, 6.0])
    edges, features = family_edges(frame, k=2)
    new_edges, new_features = family_edges(frame.assign(gt_track_id=[4, 1, 3, 2]), k=2)
    np.testing.assert_array_equal(edges, new_edges)
    np.testing.assert_allclose(features, new_features)


def test_collective_decoder_matches_nearest_without_affinity_and_obeys_capacity():
    cost = np.array([[0.0, 3.0], [0.1, 2.0], [0.2, 1.0], [0.3, 0.0]])
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    affinity = np.array([0.9, 0.9, 0.9])
    np.testing.assert_array_equal(collective_assignment(cost, edges, affinity, 0, 0.5, 1),
                                  cost.argmin(1))
    assignment = collective_assignment(cost, edges, affinity, 0, 0.5, 1, capacity=2)
    assert np.bincount(assignment, minlength=2).max() <= 2
