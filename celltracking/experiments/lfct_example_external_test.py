"""Locked zero-shot OIAM -> LFCT FLD_3 example endpoint ancestry test.

Protocol: docs/261002_lfct_example_external_protocol.md. TrackMate spot IDs
and edges are evaluator-only; inference features use endpoint coordinates,
population context and elapsed minutes. This is one LFCT example field, not
the full LFCT benchmark.

Run from celltracking/: python experiments/lfct_example_external_test.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402
from experiments.pairwise_geometry_baseline import (HORIZONS, pair_assignment,
                                                    train_classifier)  # noqa: E402

XML = ROOT / "data/raw/lfct_example/FLD_3.xml"
GAPS = (2, 4, 8)
MINUTES_PER_FRAME = 15


def load_graph(path):
    root = etree.parse(str(path))
    spots = {}
    by_frame = {}
    for spot in root.xpath("//AllSpots//Spot"):
        sid = int(spot.get("ID"))
        frame = int(spot.get("FRAME"))
        spots[sid] = (frame, float(spot.get("POSITION_Y")),
                      float(spot.get("POSITION_X")))
        by_frame.setdefault(frame, []).append(sid)
    parent = {}
    incoming = Counter()
    outgoing = Counter()
    gaps = Counter()
    for edge in root.xpath("//AllTracks//Edge"):
        src, dst = int(edge.get("SPOT_SOURCE_ID")), int(edge.get("SPOT_TARGET_ID"))
        if src not in spots or dst not in spots:
            raise ValueError("Edge references missing spot")
        if spots[src][0] >= spots[dst][0]:
            raise ValueError("Edge not forward in time")
        incoming[dst] += 1
        outgoing[src] += 1
        gaps[spots[dst][0] - spots[src][0]] += 1
        if dst in parent:
            parent[dst] = None  # Ambiguous merge, excluded by the evaluator.
        else:
            parent[dst] = src
    print("graph", {"frames": len(by_frame), "spots": len(spots),
                    "edges": sum(gaps.values()), "splits": sum(v > 1 for v in outgoing.values()),
                    "merges": sum(v > 1 for v in incoming.values()),
                    "edge_frame_gaps": dict(gaps)}, flush=True)
    return spots, by_frame, parent


def lineage_at_frame(sid, source_frame, spots, parent):
    seen = set()
    while spots[sid][0] > source_frame:
        if sid in seen:
            raise ValueError("Cycle in lineage graph")
        seen.add(sid)
        sid = parent.get(sid)
        if sid is None:
            return None
    return sid if spots[sid][0] == source_frame else None


def family_f1(truth, predicted):
    keys = set(truth.tolist()) | set(predicted.tolist())
    return float(np.mean([2 * np.sum((truth == k) & (predicted == k)) /
                          (np.sum(truth == k) + np.sum(predicted == k))
                          for k in keys]))


def make_window(start, gap, spots, by_frame, parent, rng):
    source = rng.permutation(by_frame[start])
    target = rng.permutation(by_frame[start + gap])
    source_index = {sid: i for i, sid in enumerate(source)}
    valid_rows, truth = [], []
    for j, sid in enumerate(target):
        ancestor = lineage_at_frame(sid, start, spots, parent)
        if ancestor in source_index:
            valid_rows.append(j)
            truth.append(source_index[ancestor])
    anchor = pd.DataFrame({"local_detection_id": np.arange(len(source)),
                           "center_y": [spots[s][1] for s in source],
                           "center_x": [spots[s][2] for s in source]})
    target_df = pd.DataFrame({"local_detection_id": np.arange(len(target)),
                              "center_y": [spots[s][1] for s in target],
                              "center_x": [spots[s][2] for s in target]})
    src_xy = anchor[["center_y", "center_x"]].to_numpy()
    dst_xy = target_df[["center_y", "center_x"]].to_numpy()
    cost = ((dst_xy[:, None, :] - src_xy[None, :, :]) ** 2).sum(-1)
    e = {"anchor": anchor, "target": target_df,
         "window": {"horizon_frames": gap * MINUTES_PER_FRAME},
         "cost": cost.astype(np.float32)}
    return e, np.asarray(valid_rows, dtype=int), np.asarray(truth, dtype=int)


def main():
    spots, by_frame, parent = load_graph(XML)
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in HORIZONS
             for e in examples(alias, windows, compensate=False, horizon=h)]
    pair_model = train_classifier(train)
    center_model = fit_multi(train)
    rng = np.random.default_rng(20261002)
    rows = []
    for gap in GAPS:
        for start in sorted(by_frame):
            if start + gap not in by_frame:
                continue
            e, valid_rows, truth = make_window(start, gap, spots, by_frame,
                                               parent, rng)
            if len(valid_rows) == 0:
                continue
            assignments = {
                "nearest_anchor": e["cost"].argmin(axis=1),
                "family_center": multi_cost(e, center_model).argmin(axis=1),
                "pairwise_geometry": pair_assignment(e, pair_model),
            }
            for name, assignment in assignments.items():
                guess = assignment[valid_rows]
                rows.append({"sequence_id": "FLD_3", "window_id": f"{start}_{start+gap}",
                             "source_frame": start, "target_frame": start + gap,
                             "gap_frames": gap, "gap_minutes": gap * MINUTES_PER_FRAME,
                             "n_anchors": len(e["anchor"]), "n_targets": len(e["target"]),
                             "n_scored": len(truth),
                             "n_no_source_ancestor": len(e["target"]) - len(truth),
                             "growth_ratio": len(e["target"]) / len(e["anchor"]),
                             "method": name, "accuracy": float(np.mean(guess == truth)),
                             "set_f1": family_f1(truth, guess)})
        print("gap", gap, "complete", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "lfct_example_external_test.parquet"
    out.to_parquet(path, index=False)
    print(out.groupby(["gap_minutes", "method"])[["accuracy", "set_f1"]].mean().round(4), flush=True)
    print(out[out.method == "nearest_anchor"].groupby("gap_minutes")
          .agg(n_windows=("window_id", "size"), n_scored=("n_scored", "sum"),
               n_excluded=("n_no_source_ancestor", "sum"),
               anchor_median=("n_anchors", "median"),
               growth_median=("growth_ratio", "median")).round(3), flush=True)
    pair = out.pivot(index=["window_id", "gap_minutes"], columns="method", values="accuracy")
    for gap, group in pair.groupby(level="gap_minutes"):
        for name in ("family_center", "pairwise_geometry"):
            diff = group[name] - group.nearest_anchor
            print("wins/ties/losses vs nearest", gap, name,
                  int((diff > 1e-12).sum()), int((diff.abs() <= 1e-12).sum()),
                  int((diff < -1e-12).sum()), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
