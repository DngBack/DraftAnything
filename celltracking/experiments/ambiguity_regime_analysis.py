"""Label-free endpoint ambiguity and observed method gains across development sets.

This is post-hoc hypothesis analysis. OIAM, FAST and LFCT have all been
viewed; any rule chosen here requires a new prospective test collection.

Run from celltracking/: python experiments/ambiguity_regime_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import ALIASES, BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.fast_microcolony_external_test import GAPS, HELD_OUT, load_colony  # noqa: E402
from experiments.lfct_example_external_test import XML, load_graph  # noqa: E402


def ambiguity(source, target):
    if len(source) > 1:
        ds, _ = cKDTree(source).query(source, k=2)
        spacing = max(float(np.median(ds[:, 1])), 1e-6)
        dt, _ = cKDTree(source).query(target, k=2)
        dt = np.asarray(dt).reshape(len(target), 2)
        d1, d2 = dt[:, 0], dt[:, 1]
        margin = (d2 - d1) / spacing
        ratio = d1 / np.maximum(d2, 1e-6)
    else:
        spacing = 1.0
        d1 = np.linalg.norm(target - source[0], axis=1)
        margin = np.full(len(target), np.inf)
        ratio = np.zeros(len(target))
    return {"n_anchors": len(source), "n_targets": len(target),
            "growth_ratio": len(target) / len(source),
            "source_spacing": spacing,
            "nearest_distance_scaled": float(np.median(d1 / spacing)),
            "ambiguity_ratio_median": float(np.median(ratio)),
            "ambiguity_margin_median": float(np.median(margin)),
            "ambiguous_fraction": float(np.mean(margin < 0.5))}


def oiam_rows():
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ALIASES:
        for gap in (33, 66, 132):
            for e in examples(alias, windows, compensate=False, horizon=gap):
                src = e["anchor"][["center_y", "center_x"]].to_numpy()
                dst = e["target"][["center_y", "center_x"]].to_numpy()
                rows.append({"dataset": "OIAM", "sequence": alias,
                             "window_id": e["window"]["window_id"],
                             "horizon_min": gap, **ambiguity(src, dst)})
    return rows


def fast_rows():
    rows = []
    for alias in HELD_OUT:
        frames, _, _, _ = load_colony(ROOT / "data/downloads" / f"fast_{alias}.zip")
        for gap in GAPS:
            for start in sorted(frames):
                if start + gap in frames:
                    rows.append({"dataset": "FAST", "sequence": alias,
                                 "window_id": f"{start}_{start+gap}",
                                 "horizon_min": gap * 5,
                                 **ambiguity(frames[start], frames[start + gap])})
        print("FAST", alias, "done", flush=True)
    return rows


def lfct_rows():
    spots, by_frame, _ = load_graph(XML)
    rows = []
    for gap in (2, 4, 8):
        for start in sorted(by_frame):
            if start + gap in by_frame:
                a = np.asarray([[spots[s][1], spots[s][2]] for s in by_frame[start]])
                b = np.asarray([[spots[s][1], spots[s][2]] for s in by_frame[start + gap]])
                rows.append({"dataset": "LFCT", "sequence": "FLD_3",
                             "window_id": f"{start}_{start+gap}",
                             "horizon_min": gap * 15, **ambiguity(a, b)})
    return rows


def metrics():
    o = pd.read_parquet(RESULTS / "leave_one_movie_pairwise_control_all.parquet")
    o["accuracy"] = o.n_correct / o.n_scored
    o = o.rename(columns={"sequence_id": "sequence", "horizon_frames": "horizon_min"})
    o["dataset"] = "OIAM"
    f = pd.read_parquet(RESULTS / "fast_microcolony_external_test.parquet")
    f = f.rename(columns={"colony": "sequence", "gap_minutes_assumed": "horizon_min"})
    f["dataset"] = "FAST"
    l = pd.read_parquet(RESULTS / "lfct_example_external_test.parquet")
    l = l.rename(columns={"sequence_id": "sequence", "gap_minutes": "horizon_min"})
    l["dataset"] = "LFCT"
    out = pd.concat([o, f, l], ignore_index=True)
    return out.pivot(index=["dataset", "sequence", "window_id", "horizon_min"],
                     columns="method", values="accuracy").reset_index()


def main():
    x = pd.DataFrame(oiam_rows() + fast_rows() + lfct_rows())
    x = x.merge(metrics(), on=["dataset", "sequence", "window_id", "horizon_min"],
                how="inner", validate="one_to_one")
    x["center_minus_pair"] = x.family_center - x.pairwise_geometry
    x["center_minus_nearest"] = x.family_center - x.nearest_anchor
    x["pair_minus_nearest"] = x.pairwise_geometry - x.nearest_anchor
    path = RESULTS / "ambiguity_regime_analysis.parquet"
    x.to_parquet(path, index=False)
    print("saved", len(x), path, flush=True)
    for dataset, part in x.groupby("dataset"):
        print("\n", dataset, "medians", part[["n_anchors", "growth_ratio",
              "ambiguity_ratio_median", "ambiguous_fraction",
              "nearest_distance_scaled"]].median().round(3).to_dict(), flush=True)
        print("Spearman vs gains", part[["n_anchors", "growth_ratio",
              "ambiguity_ratio_median", "ambiguous_fraction",
              "nearest_distance_scaled", "center_minus_pair",
              "center_minus_nearest"]].corr(method="spearman").loc[
              ["n_anchors", "growth_ratio", "ambiguity_ratio_median",
               "ambiguous_fraction", "nearest_distance_scaled"],
              ["center_minus_pair", "center_minus_nearest"]].round(3).to_string(),
              flush=True)


if __name__ == "__main__":
    main()
