"""Does OT decode beat LAP decode on a completely handcrafted, non-learned cost (raw pixel
distance after expansion compensation, uniform mass -- no shape/area needed) across datasets?

This isolates the "OT decode as a universal, cost-agnostic upgrade" question (doc: 260930
synthesis, sec 6/Next Actions) from the learned-embedding-cost question eval_embed.py answers.
Config (uniform mass, eps=0.1, tau=3.0, soft) is the best found on OIAM in
outputs/results/decide_h132_tune.parquet; reused as-is on other datasets, no re-tuning, since
tuning per-target-domain would itself need labels the point is to avoid depending on.

  uv run python experiments/ot_transfer.py --dataset musc --seqs MuSC-01 MuSC-02 --H 110
    -> outputs/results/ot_transfer_musc_h110.parquet
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import BASELINES  # noqa: E402
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from cellanc.transport import ot_ancestry  # noqa: E402
from main import BENCH, DATA, DATASETS  # noqa: E402

EPS, TAU = 0.1, 3.0
BASELINE_NAMES = ["chained_lap_c4", "chained_lap_rc", "nearest_anchor_rc"]
_WS = Windows(BENCH, DATA)
_tg_cache: dict = {}


def run(w: dict) -> list[dict]:
    win = _WS.load(w)
    sib = pd.DataFrame(columns=["det_1", "det_2"])
    if w["sequence_id"] not in _tg_cache:
        _tg_cache[w["sequence_id"]] = pd.read_parquet(
            BENCH / "labels/targets" / f"{w['sequence_id']}.parquet")
    tg = _tg_cache[w["sequence_id"]]
    tg = tg[tg.window_id == w["window_id"]]
    mass = {t: np.ones(len(d)) for t, d in win["detections"].items()}
    preds = {b: BASELINES[b](win) for b in BASELINE_NAMES}
    preds[f"ot_uniform_e{EPS}_t{TAU}"] = ot_ancestry(win, mass, EPS, TAU, soft=True)
    rows = []
    for name, p in preds.items():
        s = score_window(p, set(), tg, sib)
        rows.append({"method": name, "window_id": w["window_id"], "sequence_id": w["sequence_id"],
                    "schedule_id": w["schedule_id"], "horizon_frames": w["horizon_frames"],
                    **{k: s[k] for k in ["n_scored", "n_correct", "set_f1_sum", "count_n_anchors"]}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="one_in_a_million", choices=DATASETS)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--schedules", nargs="+", default=["ends", "s16", "s32"])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    ws = [w for w in _WS.windows if w["sequence_id"] in args.seqs
          and w["horizon_frames"] == args.H and w["schedule_id"] in args.schedules
          and f"A_h{args.H}" in w["experiments"]]
    print(len(ws), "windows", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        R = pd.DataFrame([r for rows in ex.map(run, ws) for r in rows])
    dataset_tag = "" if args.dataset == "one_in_a_million" else f"_{args.dataset}"
    out = Path(__file__).parents[1] / "outputs/results" / f"ot_transfer{dataset_tag}_h{args.H}.parquet"
    R.to_parquet(out, index=False)
    R["acc"] = R.n_correct / R.n_scored.replace(0, np.nan)
    print(R.groupby(["method", "schedule_id"]).acc.mean().unstack().round(3))
    print("saved", out)


if __name__ == "__main__":
    main()
