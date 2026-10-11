"""Zero-shot OIAM-to-CTC transfer of the horizon-conditioned center prior.

Train the point-cloud center regressor on OIAM M0-M2 at 33/66/132 minutes.
Evaluate nearest-anchor versus nearest predicted family center on CTC oracle
locations at approximately 116-160 minutes. No CTC labels are used to fit or
select the model. This isolates registration transfer; no family affinity or
CTC shape information is used, and there are few labeled windows per movie.

Run from celltracking/: python experiments/ctc_center_transfer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.family import endpoint_cost  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402


HORIZONS = {"HSC": 32, "MuSC": 32, "HeLa": 4, "SIM": 4}


def select_windows(all_windows):
    return [w for w in all_windows if w["dataset"] in ("hsc", "musc", "hela", "sim_plus")
            and w["schedule_id"] == "ends"
            and w["horizon_frames"] == HORIZONS[w["sequence_id"].split("-")[0]]]


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in (33, 66, 132)
             for e in examples(alias, windows, compensate=False, horizon=h)]
    model = fit_multi(train)
    selected = select_windows(windows)
    empty = pd.DataFrame(columns=["det_1", "det_2"])
    rows = []
    for alias in sorted({w["sequence_id"] for w in selected}):
        det = pd.read_parquet(BENCH / "inputs/detections" / f"{alias}.parquet")
        by_frame = {int(t): g.sort_values("local_detection_id").reset_index(drop=True)
                    for t, g in det.groupby("frame_index")}
        targets = dict(list(pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
                            .groupby("window_id")))
        for w in (w for w in selected if w["sequence_id"] == alias):
            anchor, target = by_frame[w["anchor_frame"]], by_frame[w["target_frame"]]
            if not len(anchor) or not len(target):
                continue
            e = {"anchor": anchor, "target": target,
                 "window": {**w, "horizon_frames": w["horizon_min"]}}
            costs = {"nearest_anchor": endpoint_cost(anchor, target, compensate=False),
                     "oiam_center_transfer": multi_cost(e, model)}
            for name, c in costs.items():
                ids = anchor.local_detection_id.to_numpy(dtype=int)[c.argmin(axis=1)]
                pred = dict(zip(target.local_detection_id.to_numpy(dtype=int), ids))
                s = score_window(pred, set(), targets[w["window_id"]], empty)
                rows.append({"sequence_id": alias, "window_id": w["window_id"],
                             "horizon_min": w["horizon_min"], "method": name,
                             "n_anchors": len(anchor), "n_scored": s["n_scored"],
                             "n_correct": s["n_correct"],
                             "set_f1_sum": s["set_f1_sum"],
                             "count_n_anchors": s["count_n_anchors"]})
        print(alias, "complete", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "ctc_center_transfer.parquet"
    out.to_parquet(path, index=False)
    out = out[out.n_scored > 0].copy()
    out["acc"] = out.n_correct / out.n_scored
    out["f1"] = out.set_f1_sum / out.count_n_anchors
    print(out.groupby(["sequence_id", "method"])[["acc", "f1"]].agg(["mean", "count"]).round(4).to_string())
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
