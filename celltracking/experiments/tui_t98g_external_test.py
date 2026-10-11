"""Predeclared zero-shot T98G electrotaxis endpoint ancestry test.

Protocol: docs/261002_tui_external_protocol.md, written before downloading GT.
Human-curated TRA masks are reduced to per-frame centroids and assigned fresh
local IDs; original track IDs and man_track.txt are used only by the evaluator.
The model and hyperparameters are imported unchanged from growth_clock_transfer.

Run from celltracking/: python experiments/tui_t98g_external_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.family import endpoint_cost  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.growth_clock_transfer import TRAIN_H, fit, predicted_cost  # noqa: E402


MASK_DIR = ROOT / "data/raw/tui/T98G_electrotaxis/T98G_human/T98G_sample_GT/TRA"
GAPS = {33: 3, 66: 7, 132: 13}  # 10 min/frame, closest within 20% tolerance


def frame_table(path: Path, frame_index: int) -> tuple[pd.DataFrame, np.ndarray]:
    mask = tifffile.imread(path)
    yy, xx = np.nonzero(mask)
    labels = mask[yy, xx].astype(int)
    count = np.bincount(labels)
    ids = np.flatnonzero(count)[1:] if count[0] else np.flatnonzero(count)
    sy = np.bincount(labels, weights=yy, minlength=len(count))
    sx = np.bincount(labels, weights=xx, minlength=len(count))
    rng = np.random.default_rng(20261002 + frame_index)
    permutation = rng.permutation(len(ids))
    ids = ids[permutation]
    data = pd.DataFrame({"local_detection_id": np.arange(1, len(ids) + 1),
                         "center_y": sy[ids] / count[ids],
                         "center_x": sx[ids] / count[ids]})
    return data, ids


def lineage(path: Path) -> dict[int, int]:
    rows = np.loadtxt(path, dtype=int).reshape(-1, 4)
    return {int(track): int(parent) for track, _, _, parent in rows}


def target_labels(source_ids: np.ndarray, target_ids: np.ndarray,
                  parent: dict[int, int]) -> tuple[pd.DataFrame, int]:
    source = {int(track): i + 1 for i, track in enumerate(source_ids)}
    rows = []
    unresolved = 0
    for target_local, track in enumerate(target_ids, 1):
        node = int(track)
        depth = 0
        visited = set()
        while node not in source and node != 0 and node not in visited:
            visited.add(node)
            node = parent.get(node, 0)
            depth += 1
        ancestor = source.get(node)
        if ancestor is None:
            unresolved += 1
        rows.append({"target_detection_id": target_local,
                     "ancestor_detection_id": ancestor,
                     "target_status": "valid_anchor" if ancestor is not None else "new_root",
                     "generation_depth": depth if ancestor is not None else 0,
                     "hidden_intermediates": depth > 0 if ancestor is not None else False,
                     "in_foi": True})
    return pd.DataFrame(rows), unresolved


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in TRAIN_H
             for e in examples(alias, windows, compensate=False, horizon=h,
                               require_experiment=h in (33, 66, 132))]
    models = {name: fit(train, name) for name in ("physical_time", "growth_clock")}
    parent = lineage(MASK_DIR / "man_track.txt")
    frames = {t: frame_table(MASK_DIR / f"mask{t:03}.tif", t) for t in range(37)}
    all_ids = {int(x) for _, ids in frames.values() for x in ids}
    if not all_ids.issubset(parent):
        raise ValueError(f"mask IDs missing from man_track: {sorted(all_ids - parent.keys())[:10]}")
    empty = pd.DataFrame(columns=["det_1", "det_2"])
    rows = []
    for intended, gap in GAPS.items():
        minutes = gap * 10
        for a in range(37 - gap):
            b = a + gap
            src, src_gt = frames[a]
            dst, dst_gt = frames[b]
            truth, unresolved = target_labels(src_gt, dst_gt, parent)
            e = {"anchor": src, "target": dst,
                 "window": {"horizon_min": minutes}}
            costs = {"nearest_anchor": endpoint_cost(src, dst, compensate=False)}
            costs.update({name: predicted_cost(e, model, name)
                          for name, model in models.items()})
            for name, cost in costs.items():
                anc = src.local_detection_id.to_numpy(dtype=int)[cost.argmin(axis=1)]
                pred = dict(zip(dst.local_detection_id.to_numpy(dtype=int), anc))
                score = score_window(pred, set(), truth, empty)
                rows.append({"sequence_id": "T98G", "window_id": f"T98G_a{a:03}_b{b:03}",
                             "intended_horizon_min": intended, "horizon_min": minutes,
                             "method": name, "n_anchors": len(src), "n_targets": len(dst),
                             "n_unresolved": unresolved, "n_scored": score["n_scored"],
                             "n_correct": score["n_correct"],
                             "set_f1_sum": score["set_f1_sum"],
                             "count_n_anchors": score["count_n_anchors"]})
        print("gap", gap, "windows", 37 - gap, "complete", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "tui_t98g_external_test.parquet"
    out.to_parquet(path, index=False)
    out["acc"] = out.n_correct / out.n_scored
    out["f1"] = out.set_f1_sum / out.count_n_anchors
    print("macro", out.groupby(["horizon_min", "method"])[["acc", "f1"]].mean().round(4).to_string(), flush=True)
    print("pooled", out.groupby("method").apply(
        lambda g: pd.Series({"accuracy": g.n_correct.sum() / g.n_scored.sum(),
                             "set_f1": g.set_f1_sum.sum() / g.count_n_anchors.sum(),
                             "n_scored": g.n_scored.sum(),
                             "n_unresolved": g.n_unresolved.sum()}),
        include_groups=False).round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
