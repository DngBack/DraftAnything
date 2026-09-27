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


def test_lap_capacity_four_and_fallback_is_reported():
    # 1 parent, 4 children: cap 2 is infeasible -> nearest + reported; cap 4 solves it
    from cellanc.baselines import BASELINES
    det = {0: np.array([[1, 0.0, 0.0], [2, 0.0, 50.0]]),
           1: np.array([[i, 0.0, x] for i, x in enumerate([-2.0, -1.0, 1.0, 2.0, 49.0], 1)])}
    det[1] = np.vstack([det[1], [[6, 0.0, 51.0]]])
    win = _win(det)
    assert BASELINES["chained_lap_c4"](win) == {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2}
    assert win["link_stats"] == (0, 1)
    det[1] = np.vstack([det[1], [[7, 0.0, 3.0], [8, 0.0, 4.0]]])  # 8 children > 2 * 2 parents
    win = _win(det)
    chained_lap(win)
    assert win["link_stats"] == (1, 1)
    BASELINES["chained_lap_soft"](win)
    assert win["link_stats"] == (0, 1)


def test_set_f1_catches_swapped_families():
    from cellanc.evaluate import score_window
    tg = pd.DataFrame({"target_detection_id": [1, 2, 3, 4], "ancestor_detection_id": [10, 10, 20, 20],
                       "target_status": "valid_anchor", "in_foi": True, "generation_depth": 1,
                       "hidden_intermediates": 0})
    sib = pd.DataFrame(columns=["det_1", "det_2"])
    s = score_window({1: 20, 2: 20, 3: 10, 4: 10}, set(), tg, sib)
    assert s["count_abs_err_sum"] == 0 and s["set_f1_sum"] == 0 and s["count_abs_err_gt_sum"] == 0
    s = score_window({1: 10, 2: 10, 3: 20, 4: 10}, set(), tg, sib)
    assert s["set_f1_sum"] == 2 * 2 / 5 + 2 * 1 / 3


def test_matched_budget_schedules():
    s = schedules_for(0, 132, 0)
    for n in (3, 5, 9, 17):
        assert len(s[f"u{n}"]) == n and all(len(s[f"r{n}_{j}"]) == n for j in range(5))
        assert s[f"u{n}"][0] == 0 and s[f"u{n}"][-1] == 132
    assert len({tuple(s[f"r5_{j}"]) for j in range(5)}) == 5


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

    root = Path(__file__).parents[1] / "outputs/benchmarks/v1/inputs"
    if not root.exists():
        pytest.skip("benchmark not built")
    banned = {"gt_track_id", "parent_id", "generation_depth", "ancestor_detection_id"}
    for p in (root / "detections").glob("*.parquet"):
        assert set(pd.read_parquet(p).columns) == {"frame_index", "local_detection_id",
                                                   "center_y", "center_x"}
    for p in (root / "windows").glob("*.jsonl"):
        for line in p.read_text().splitlines():
            assert not banned & set(json.loads(line))


def test_mass_transport_sends_extra_child_to_the_bigger_anchor():
    # anchor 1 (area 2) is about to split, anchor 2 (area 1) is not. Child 3 at x=6 is nearer to
    # anchor 2, but anchor 2's mass is used up by child 2 at x=11, so mass sends child 3 to 1.
    from cellanc.transport import ot_ancestry
    det = {0: np.array([[1, 0.0, 0.0], [2, 0.0, 10.0]]),
           1: np.array([[1, 0.0, -1.0], [2, 0.0, 11.0], [3, 0.0, 6.0]])}
    mass = {0: np.array([2.0, 1.0]), 1: np.ones(3)}
    win = _win(det)
    assert nearest_anchor(win)[3] == 2
    assert ot_ancestry(win, mass, eps=0.05, tau=None, soft=True) == {1: 1, 2: 2, 3: 1}


def test_anisotropic_cost_follows_the_long_axis():
    # target 1 lies straight along y from anchor 2, but anchor 2 is a rod along x while anchor 1
    # is a rod along y; making along-axis moves cheap hands target 1 to anchor 1.
    from cellanc.transport import ot_ancestry
    det = {0: np.array([[1, 0.0, 0.0], [2, 0.0, 10.0]]),
           1: np.array([[1, 15.0, 10.0], [2, 6.0, 0.0], [3, -2.0, 15.0]])}
    mass, axis = {0: np.ones(2), 1: np.ones(3)}, {0: np.array([[1.0, 0.0], [0.0, 1.0]])}
    win = _win(det)
    assert ot_ancestry(win, mass, 0.05, None, True)[1] == 2
    assert ot_ancestry(win, mass, 0.05, None, True, axis, 8.0)[1] == 1
