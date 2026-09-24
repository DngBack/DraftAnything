"""Hand-checked cases from doc §3, §9, §15 (synthetic IDs, not dataset samples)."""

import numpy as np
import pandas as pd

from cellanc.baselines import (chained_lap, chained_nearest, expansion_compensate,
                               nearest_anchor)
from cellanc.benchmark import relabel, schedules_for
from cellanc.lineage import ancestry_targets, schedule, sibling_pairs, trace_to_anchor

# doc §3 example: 1 -> (2, 3), 2 -> (4, 5), 8 independent
T = {1: (0, 4, 0), 2: (5, 9, 1), 3: (5, 12, 1), 4: (10, 15, 2), 5: (10, 15, 2), 8: (0, 15, 0)}
PRESENT = {2: {1, 8}, 12: {3, 4, 5, 8}}


def test_doc_example_table():
    rows = {r["target_gt"]: r for r in ancestry_targets(2, 12, T, PRESENT)}
    assert {u: (r["anchor_gt"], r["generation_depth"]) for u, r in rows.items()} == {
        3: (1, 1), 4: (1, 2), 5: (1, 2), 8: (8, 0)}  # 8 = ancestor-or-self
    assert all(r["status"] == "valid_anchor" for r in rows.values())


def test_siblings_direct_only():
    assert sibling_pairs(12, T, PRESENT[12]) == [(4, 5)]  # 3 is aunt of 4/5, not sister


def test_p0_is_never_a_sibling_cue():
    t = {1: (0, 5, 0), 2: (0, 5, 0)}
    assert sibling_pairs(3, t, {1, 2}) == []


def test_unknown_statuses_not_negative():
    assert trace_to_anchor(8, 2, T, {1}) == (None, "missing_anchor_annotation", 0)
    assert trace_to_anchor(99, 2, T, {1, 8})[1] == "invalid_lineage"
    t = {**T, 9: (5, 6, 0)}
    assert trace_to_anchor(9, 2, t, {1, 8})[1] == "unresolved_root_after_anchor"
    t = {1: (0, 1, 0), 2: (4, 9, 1)}  # a=2 falls in a gap between parent and child
    assert trace_to_anchor(2, 2, t, {1})[1] == "broken_temporal_path"


def test_gap_closing_link_is_not_a_generation():
    t = {1: (0, 3, 0), 2: (6, 9, 1), 3: (10, 12, 2), 4: (10, 12, 2)}  # 1 -gap-> 2 -> (3, 4)
    rows = {r["target_gt"]: r for r in ancestry_targets(1, 11, t, {1: {1}, 11: {3, 4}})}
    assert rows[3]["anchor_gt"] == 1 and rows[3]["generation_depth"] == 1


def test_hidden_intermediates_depend_on_schedule():
    dense = ancestry_targets(2, 12, T, PRESENT, list(range(2, 13)))
    ends = ancestry_targets(2, 12, T, PRESENT, [2, 12])
    assert {r["target_gt"]: r["hidden_intermediates"] for r in dense}[4] == 0
    assert {r["target_gt"]: r["hidden_intermediates"] for r in ends}[4] == 1


def test_schedules_keep_endpoints():
    assert schedule(20, 60, 8) == [20, 28, 36, 44, 52, 60]
    assert schedule(20, 62, 8)[-1] == 62
    for fr in schedules_for(20, 60, 0).values():
        assert fr[0] == 20 and fr[-1] == 60 and fr == sorted(set(fr))


def test_relabel_is_permutation_and_hides_gt_order():
    obs = pd.DataFrame({"frame_index": np.repeat([0, 1], 500), "gt_track_id": np.tile(
        np.arange(1, 501), 2), "center_y": 0.0, "center_x": 0.0})
    loc = relabel(obs, "X", 0)
    for _, g in loc.groupby("frame_index"):
        assert sorted(g.local_detection_id) == list(range(1, 501))
    assert abs(np.corrcoef(loc.gt_track_id, loc.local_detection_id)[0, 1]) < 0.1
    # same cell gets unrelated local ids in different frames
    f0 = loc[loc.frame_index == 0].set_index("gt_track_id").local_detection_id
    f1 = loc[loc.frame_index == 1].set_index("gt_track_id").local_detection_id
    assert (f0 == f1.reindex(f0.index)).mean() < 0.05


def test_baselines_on_toy_division():
    det = {0: np.array([[1, 0.0, 0.0], [2, 100.0, 100.0]]),
           5: np.array([[1, 3.0, 0.0], [2, -3.0, 0.0], [3, 100.0, 101.0]])}
    win = {"window": {"anchor_frame": 0, "target_frame": 5, "observed_frames": [0, 5], "sequence_id": "toy"},
           "detections": det}
    assert nearest_anchor(win) == {1: 1, 2: 1, 3: 2}
    assert chained_nearest(win) == {1: 1, 2: 1, 3: 2}
    assert chained_lap(win) == {1: 1, 2: 1, 3: 2}


def _win(det):
    fr = sorted(det)
    return {"window": {"anchor_frame": fr[0], "target_frame": fr[-1], "observed_frames": fr,
                       "sequence_id": f"toy{id(det)}"}, "detections": det}


def test_lap_caps_children_at_two():
    # parent 1 sits closest to all three children; nearest gives it 3, capacity 2 sends one to 2
    det = {0: np.array([[1, 0.0, 0.0], [2, 0.0, 10.0]]),
           1: np.array([[1, 0.0, -1.0], [2, 0.0, 1.0], [3, 0.0, 3.0]])}
    assert sorted(chained_nearest(_win(det)).values()) == [1, 1, 1]
    assert sorted(chained_lap(_win(det)).values()) == [1, 1, 2]


def test_expansion_compensation_undoes_radial_growth():
    src = np.array([[1, 0.0, 0.0], [2, 10.0, 0.0], [3, 0.0, 10.0], [4, 10.0, 10.0]])
    dst = src.copy()
    dst[:, 1:] = (src[:, 1:] - 5) * 1.8 + 5
    assert np.allclose(expansion_compensate(src, dst), src)


def test_built_inputs_have_no_gt_fields():
    """Leakage check on the real benchmark output, if it has been built (doc §15)."""
    import json
    from pathlib import Path

    import pytest

    root = Path(__file__).parents[1] / "outputs/benchmarks/v0/inputs"
    if not root.exists():
        pytest.skip("benchmark not built")
    banned = {"gt_track_id", "parent_id", "generation_depth", "ancestor_detection_id"}
    for p in (root / "detections").glob("*.parquet"):
        assert set(pd.read_parquet(p).columns) == {"frame_index", "local_detection_id",
                                                   "center_y", "center_x"}
    for p in (root / "windows").glob("*.jsonl"):
        for line in p.read_text().splitlines():
            assert not banned & set(json.loads(line))
