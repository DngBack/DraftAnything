"""Direct nearest-center control without family affinity or joint decoder.

Train pooled and horizon-conditioned center regressors on M0-M2 at H=33/66/132;
evaluate M3/M4 by simple nearest assignment for all methods. Oracle endpoint
locations are the only inference input. M4 was previously inspected.

Run from celltracking/: python experiments/nearest_center_horizon_control.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_center_information_ablation import SOURCE  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402


HORIZONS = (33, 66, 132)


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in HORIZONS
             for e in examples(alias, windows, compensate=False, horizon=h)]
    pooled = fit_multi(train, include_horizon=False)
    multi = fit_multi(train, include_horizon=True)
    source = fit_multi(train, include_horizon=True, indices=SOURCE)
    empty = pd.DataFrame(columns=["det_1", "det_2"])
    rows = []
    for alias in ("M3", "M4"):
        for h in HORIZONS:
            items = examples(alias, windows, compensate=False, horizon=h)
            for e in items:
                costs = {"nearest_anchor": e["cost"],
                         "pooled_no_horizon": multi_cost(e, pooled, False),
                         "multi_horizon": multi_cost(e, multi, True),
                         "source_time_prior": multi_cost(e, source, True, SOURCE)}
                for name, c in costs.items():
                    aid = e["anchor"].local_detection_id.to_numpy(dtype=int)
                    tid = e["target"].local_detection_id.to_numpy(dtype=int)
                    pred = dict(zip(tid, aid[c.argmin(axis=1)]))
                    s = score_window(pred, set(), e["targets"], empty)
                    rows.append({"sequence_id": alias, "window_id": e["window"]["window_id"],
                                 "horizon_frames": h, "method": name,
                                 "n_anchors": len(aid), "n_scored": s["n_scored"],
                                 "n_correct": s["n_correct"],
                                 "set_f1_sum": s["set_f1_sum"],
                                 "count_n_anchors": s["count_n_anchors"]})
            print(alias, h, "complete", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "nearest_center_horizon_control.parquet"
    out.to_parquet(path, index=False)
    for label, data in (("all", out), ("dense", out[out.n_anchors > 200])):
        acc = (data.n_correct / data.n_scored).groupby([data.horizon_frames, data.method]).mean()
        f1 = (data.set_f1_sum / data.count_n_anchors).groupby([data.horizon_frames, data.method]).mean()
        print(label, pd.DataFrame({"acc": acc, "f1": f1}).round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
