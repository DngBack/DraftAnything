"""Exploratory endpoint geometry for outward-growing bacterial colonies.

recA and FAST labels were viewed before this probe. This is development
analysis, not confirmation. No lineage information enters inference.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.fast_microcolony_external_test import (  # noqa: E402
    GAPS as FAST_GAPS, HELD_OUT as FAST_HELD_OUT,
    family_f1, load_colony as load_fast, make_window as fast_window)
from experiments.reca_microcolony_external_test import (  # noqa: E402
    GAPS as RECA_GAPS, HELD_OUT as RECA_HELD_OUT,
    make_window as reca_window)
from experiments.reca_schnitzcells import load_colony as load_reca  # noqa: E402
from experiments.lfct_example_external_test import (  # noqa: E402
    XML, load_graph, make_window as lfct_window)
from experiments.pairwise_geometry_baseline import HORIZONS, score  # noqa: E402
from experiments.tui_t98g_external_test import (  # noqa: E402
    GAPS as T98G_GAPS, MASK_DIR, frame_table, lineage, target_labels)


def geometry_assignments(e):
    source = e["anchor"][["center_y", "center_x"]].to_numpy(dtype=float)
    target = e["target"][["center_y", "center_x"]].to_numpy(dtype=float)
    if len(source) == 1:
        return {name: np.zeros(len(target), dtype=int)
                for name in ("nearest_anchor", "radial_sector",
                             "radial_expansion", "centroid_translation")}
    center_s, center_t = source.mean(axis=0), target.mean(axis=0)
    vs, vt = source - center_s, target - center_t
    norm_s = np.linalg.norm(vs, axis=1)
    norm_t = np.linalg.norm(vt, axis=1)
    cosine = (vt @ vs.T) / np.maximum(norm_t[:, None] * norm_s[None, :], 1e-12)
    sector_cost = 1 - cosine
    scale = np.sqrt(len(target) / len(source))
    radial_cost = np.sum((vt[:, None, :] - scale * vs[None, :, :]) ** 2, axis=-1)
    translated_cost = np.sum((vt[:, None, :] - vs[None, :, :]) ** 2, axis=-1)
    return {"nearest_anchor": e["cost"].argmin(axis=1),
            "radial_sector": sector_cost.argmin(axis=1),
            "radial_expansion": radial_cost.argmin(axis=1),
            "centroid_translation": translated_cost.argmin(axis=1)}


def append(rows, dataset, sequence, window_id, gap, e, valid, truth):
    if not len(truth):
        return
    for method, assignment in geometry_assignments(e).items():
        guess = assignment[valid]
        rows.append({"dataset": dataset, "sequence": sequence,
                     "window_id": window_id, "gap_minutes": gap,
                     "n_anchors": len(e["anchor"]),
                     "n_targets": len(e["target"]), "n_scored": len(truth),
                     "method": method,
                     "accuracy": float(np.mean(guess == truth)),
                     "set_f1": family_f1(truth, guess)})


def main():
    rng = np.random.default_rng(20261002)
    rows = []
    windows = Windows(BENCH, ROOT / "data").windows
    for alias in ("M3", "M4"):
        for gap in HORIZONS:
            for e in examples(alias, windows, compensate=False, horizon=gap):
                for method, assignment in geometry_assignments(e).items():
                    result = score(e, assignment)
                    rows.append({"dataset": "OIAM", "sequence": alias,
                                 "window_id": e["window"]["window_id"],
                                 "gap_minutes": gap, "n_anchors": len(e["anchor"]),
                                 "n_targets": len(e["target"]),
                                 "n_scored": result["n_scored"], "method": method,
                                 "accuracy": result["n_correct"] / result["n_scored"],
                                 "set_f1": result["set_f1_sum"] / result["count_n_anchors"]})
        print("OIAM", alias, "done", flush=True)
    for name in RECA_HELD_OUT:
        frames, mapping, tracks, invalid, _ = load_reca(
            ROOT / "data/raw/reca_holdout" / name)
        for gap in RECA_GAPS:
            for start in sorted(frames):
                if start + gap not in frames:
                    continue
                e, valid, truth = reca_window(start, gap, frames, mapping,
                                               tracks, invalid, rng)
                append(rows, "recA", name, f"{start}_{start+gap}", gap * 7.5,
                       e, valid, truth)
        print("recA", name, "done", flush=True)
    for name in FAST_HELD_OUT:
        frames, mapping, tracks, _ = load_fast(
            ROOT / "data/downloads" / f"fast_{name}.zip")
        for gap in FAST_GAPS:
            for start in sorted(frames):
                if start + gap not in frames:
                    continue
                e, valid, truth = fast_window(start, gap, frames,
                                               mapping, tracks, rng)
                append(rows, "FAST", name, f"{start}_{start+gap}", gap * 5,
                       e, valid, truth)
        print("FAST", name, "done", flush=True)
    spots, by_frame, parent = load_graph(XML)
    for gap in (2, 4, 8):
        for start in sorted(by_frame):
            if start + gap not in by_frame:
                continue
            e, valid, truth = lfct_window(start, gap, spots, by_frame, parent, rng)
            append(rows, "LFCT", "FLD_3", f"{start}_{start+gap}", gap * 15,
                   e, valid, truth)
        print("LFCT", gap, "done", flush=True)
    parent = lineage(MASK_DIR / "man_track.txt")
    frames = {t: frame_table(MASK_DIR / f"mask{t:03}.tif", t) for t in range(37)}
    for _, gap in T98G_GAPS.items():
        for start in range(37 - gap):
            anchor, source_ids = frames[start]
            target, target_ids = frames[start + gap]
            labels, _ = target_labels(source_ids, target_ids, parent)
            valid_labels = labels[labels.target_status == "valid_anchor"]
            valid = valid_labels.target_detection_id.to_numpy(dtype=int) - 1
            truth = valid_labels.ancestor_detection_id.to_numpy(dtype=int) - 1
            a = anchor[["center_y", "center_x"]].to_numpy()
            b = target[["center_y", "center_x"]].to_numpy()
            cost = np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=-1)
            e = {"anchor": anchor, "target": target,
                 "window": {"horizon_frames": gap * 10},
                 "cost": cost.astype(np.float32)}
            append(rows, "T98G", "T98G", f"{start}_{start+gap}", gap * 10,
                   e, valid, truth)
        print("T98G", gap, "done", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "radial_sector_probe.parquet"
    out.to_parquet(path, index=False)
    colony = out.groupby(["dataset", "gap_minutes", "sequence", "method"])[
        ["accuracy", "set_f1"]].mean()
    print(colony.groupby(["dataset", "gap_minutes", "method"])
          .mean().round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
