"""Go/no-go experiments for the method direction (OIAM, oracle locations and masks).

E1  mass-conserving OT (area vs uniform mass), endpoints only
E2  soft (keep P(parent|child) through the schedule) vs hard (commit per link) chaining
E3  upper bound for a smooth population-motion model: displacement field target -> ancestor
    estimated by Gaussian kernel regression from the OTHER half of the targets (2-fold
    cross-fit, bandwidth h in NN spacings), then nearest anchor. Uses GT, so it is a bound,
    never a method.

  uv run python experiments/decide.py [--seqs M3 M4] [--H 132]  -> outputs/results/decide.parquet
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import BASELINES, expansion_compensate  # noqa: E402
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from cellanc.transport import ot_ancestry  # noqa: E402
from main import BENCH, DATA, OUT  # noqa: E402

SEQ = {f"M{i}": f"0{i}" for i in range(5)}
OT = [("area", 0.1, None, 1.0), ("uniform", 0.1, None, 1.0)]
H_BW = [0.5, 1, 2, 4]
_state = {}


def _load(alias):
    if alias not in _state:
        obs = pd.read_parquet(OUT / "index/one_in_a_million" / SEQ[alias] / "observations.parquet",
                              columns=["frame_index", "gt_track_id", "marker_pixels"])
        gm = pd.read_parquet(BENCH / "labels/gt_mapping" / f"{alias}.parquet")
        m = gm.merge(obs, on=["frame_index", "gt_track_id"]).merge(
            pd.read_parquet(OUT / "index/one_in_a_million" / SEQ[alias] / "shape.parquet"),
            on=["frame_index", "gt_track_id"])
        tg = pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
        _state[alias] = ({(t, i): (a, ay, ax) for t, i, a, ay, ax in zip(
                              m.frame_index, m.local_detection_id, m.marker_pixels, m.axis_y, m.axis_x)},
                         dict(list(tg.groupby("window_id"))))
    return _state[alias]


def _field_bound(win, tg, h):
    """E3: cross-fitted kernel-regression displacement field, then nearest anchor."""
    det, w = win["detections"], win["window"]
    A, B = det[w["anchor_frame"]], det[w["target_frame"]]
    ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
    pa = dict(zip(A[:, 0].astype(int), A[:, 1:]))
    pb = dict(zip(B[:, 0].astype(int), B[:, 1:]))
    x = np.array([pb[int(t)] for t in ok.target_detection_id])
    d = np.array([pa[int(a)] for a in ok.ancestor_detection_id]) - x
    bw = h * np.median(cKDTree(A[:, 1:]).query(A[:, 1:], k=2)[0][:, 1])
    fold = np.random.default_rng(0).permutation(len(x)) % 2
    pred = {}
    for k in (0, 1):
        tr, te = fold != k, fold == k
        if tr.sum() == 0 or te.sum() == 0:
            continue
        wgt = np.exp(-((x[te, None] - x[None, tr]) ** 2).sum(-1) / (2 * bw ** 2))
        disp = (wgt @ d[tr]) / np.maximum(wgt.sum(1, keepdims=True), 1e-300)
        _, idx = cKDTree(A[:, 1:]).query(x[te] + disp)
        pred.update(zip(ok.target_detection_id.to_numpy()[te].astype(int), A[idx, 0].astype(int)))
    return pred


def run(job):
    w, ot, bound = job
    alias = w["sequence_id"]
    area, tg_by = _load(alias)
    win = _WS.load(w)
    tg = tg_by[w["window_id"]]
    sib = pd.DataFrame(columns=["det_1", "det_2"])
    mass = {t: np.array([area[(t, int(i))][0] for i in d[:, 0]], float)
            for t, d in win["detections"].items()}
    axis = {t: np.array([area[(t, int(i))][1:] for i in d[:, 0]], float)
            for t, d in win["detections"].items()}
    ones = {t: np.ones(len(v)) for t, v in mass.items()}
    preds = {b: BASELINES[b](win) for b in ["nearest_anchor_rc", "chained_lap_rc", "chained_lap_c4"]}
    for mname, eps, tau, r in ot:
        for soft in ([True] if w["schedule_id"] == "ends" else [True, False]):
            preds[f"ot_{mname}_e{eps}_t{tau}" + (f"_a{r}" if r != 1 else "")
                  + f"_{'soft' if soft else 'hard'}"] = ot_ancestry(
                win, mass if mname == "area" else ones, eps, tau, soft, axis, r)
    if bound and w["schedule_id"] == "ends":
        for h in H_BW:
            preds[f"bound_field_h{h}"] = _field_bound(win, tg, h)
    rows = []
    for name, p in preds.items():
        s = score_window(p, set(), tg, sib)
        rows.append({"method": name, "window_id": w["window_id"], "sequence_id": alias,
                     "anchor_frame": w["anchor_frame"], "schedule_id": w["schedule_id"],
                     "horizon_frames": w["horizon_frames"], "n_anchors": len(win["detections"][w["anchor_frame"]]),
                     **{k: s[k] for k in ["n_scored", "n_correct", "set_f1_sum", "count_n_anchors"]}})
    return rows


_WS = Windows(BENCH, DATA)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["M3", "M4"])
    ap.add_argument("--H", type=int, default=132)
    ap.add_argument("--schedules", nargs="+", default=["ends", "s32", "s16", "u5"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--ot", nargs="+", default=None,
                    help="mass:eps:tau[:aniso], tau 'none' = balanced, aniso = along-axis cost divisor")
    ap.add_argument("--no-bound", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ws = [w for w in _WS.windows if w["sequence_id"] in args.seqs
          and w["horizon_frames"] == args.H and w["schedule_id"] in args.schedules
          and f"A_h{args.H}" in w["experiments"]]
    ws.sort(key=lambda w: -w["anchor_frame"])  # biggest windows first
    print(len(ws), "windows", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        ot = OT if args.ot is None else [(m, float(e), None if t == "none" else float(t), float(r[0]) if r else 1.0)
                                         for m, e, t, *r in (x.split(":") for x in args.ot)]
        R = pd.DataFrame([r for rows in ex.map(run, [(w, ot, not args.no_bound) for w in ws])
                          for r in rows])
    R.to_parquet(OUT / "results" / f"decide_h{args.H}{args.tag}.parquet", index=False)
    R["acc"] = R.n_correct / R.n_scored.replace(0, np.nan)
    R["f1"] = R.set_f1_sum / R.count_n_anchors.replace(0, np.nan)
    t = R.groupby(["method", "schedule_id", "sequence_id"])[["acc", "f1"]].mean().unstack("sequence_id")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
