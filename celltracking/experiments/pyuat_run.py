"""E4: PyUAT (uatrack 0.0.3, Seiffarth et al.) on the same sparse OIAM windows.

Oracle contours come from the TRA masks of the kept frames only; PyUAT gets the kept frames
renumbered 0..n-1 and subsampling_factor = the (uniform) frame gap, as in its own paper. Its
tracking graph is walked back to frame a. A cell PyUAT leaves without a parent (track start)
gives no prediction in `pyuat_<cfg>`; `pyuat_<cfg>_nn` fills that link with the nearest cell of
the previous kept frame after expansion compensation. Only uniform schedules (s*, ends).

Needs Python <= 3.10 (gurobipy wheels):
  PYTHONPATH=. <venv-3.10>/bin/python experiments/pyuat_run.py --seqs M3 --schedules ends s32
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from acia.base import Contour, Overlay
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import expansion_compensate  # noqa: E402
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs"
BENCH = OUT / "benchmarks" / "v1"
SEQ = {f"M{i}": f"0{i}" for i in range(5)}
_state = {}


def _load(alias):
    if alias not in _state:
        gm = pd.read_parquet(BENCH / "labels/gt_mapping" / f"{alias}.parquet")
        fr = pd.read_parquet(OUT / "index/one_in_a_million" / SEQ[alias] / "frames.parquet")
        tg = pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
        _state[alias] = ({(t, g): i for t, i, g in zip(gm.frame_index, gm.local_detection_id, gm.gt_track_id)},
                         dict(zip(fr.frame_index, fr.marker_path)),
                         dict(list(tg.groupby("window_id"))))
    return _state[alias]


def _overlay(frames, gid2local, marker_path):
    conts, key = [], []
    for k, t in enumerate(frames):
        m = tifffile.imread(marker_path[t])
        for g in np.unique(m[m > 0]):
            if (t, int(g)) not in gid2local:
                continue
            cs, _ = cv2.findContours((m == g).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)[:, 0, :]  # (x, y)
            if len(c) < 3:
                continue
            conts.append(Contour(c, -1, k, len(conts)))
            key.append((t, gid2local[(t, int(g))]))
    return Overlay(conts), key


def run(job):
    from uatrack.config import setup_assignment_generators
    from uatrack.core import simpleTracking
    from uatrack.utils import extract_single_cell_information

    logging.disable(logging.CRITICAL)
    w, cfgs = job
    alias = w["sequence_id"]
    gid2local, marker_path, tg_by = _load(alias)
    win = _WS.load(w)
    fr = w["observed_frames"]
    gap = fr[1] - fr[0]
    ov, key = _overlay(fr, gid2local, marker_path)
    df, _ = extract_single_cell_information(ov)
    det = win["detections"]
    rows = []
    for cfg in cfgs:
        res = simpleTracking(df, setup_assignment_generators(df, gap, cfg), 1, num_cores=1,
                             report_progress=False, max_num_hypotheses=1, cutOff=-1,
                             max_num_solutions=1, mip_method="CBC")
        parent = {key[int(b)]: key[int(a)] for a, b in res[0].tracking.createIndexTracking().edges}
        for nn in (False, True):
            par = dict(parent)
            if nn:
                for prev, cur in zip(fr, fr[1:]):
                    src, dst = det[prev], expansion_compensate(det[prev], det[cur])
                    _, idx = cKDTree(src[:, 1:]).query(dst[:, 1:])
                    for i, j in zip(det[cur][:, 0].astype(int), idx):
                        par.setdefault((cur, i), (prev, int(src[j, 0])))
            pred = {}
            for i in det[fr[-1]][:, 0].astype(int):
                node = (fr[-1], i)
                while node in par and node[0] != fr[0]:
                    node = par[node]
                if node[0] == fr[0]:
                    pred[i] = node[1]
            s = score_window(pred, set(), tg_by[w["window_id"]], pd.DataFrame(columns=["det_1", "det_2"]))
            rows.append({"method": f"pyuat_{cfg}" + ("_nn" if nn else ""), "window_id": w["window_id"],
                         "sequence_id": alias, "anchor_frame": w["anchor_frame"],
                         "schedule_id": w["schedule_id"], "horizon_frames": w["horizon_frames"],
                         "n_links": sum(1 for k in key if k[0] != fr[0]), "n_links_pyuat": len(parent),
                         **{k: s[k] for k in ["n_scored", "n_correct", "set_f1_sum", "count_n_anchors"]}})
    return rows


_WS = Windows(BENCH, ROOT / "data")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["M3"])
    ap.add_argument("--H", type=int, default=132)
    ap.add_argument("--schedules", nargs="+", default=["ends", "s32", "s16"])
    ap.add_argument("--cfgs", nargs="+", default=["FO", "FO+G+O+DD"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ws = [w for w in _WS.windows if w["sequence_id"] in args.seqs
          and w["horizon_frames"] == args.H and w["schedule_id"] in args.schedules
          and f"A_h{args.H}" in w["experiments"]]
    ws.sort(key=lambda w: -w["anchor_frame"])
    ws = ws[:args.limit]
    print(len(ws), "windows", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        R = pd.DataFrame([r for rows in ex.map(run, [(w, args.cfgs) for w in ws]) for r in rows])
    R.to_parquet(OUT / "results" / f"pyuat_h{args.H}{args.tag}.parquet", index=False)
    R["acc"] = R.n_correct / R.n_scored.replace(0, np.nan)
    R["f1"] = R.set_f1_sum / R.count_n_anchors.replace(0, np.nan)
    R["coverage"] = R.n_links_pyuat / R.n_links
    t = R.groupby(["method", "schedule_id", "sequence_id"])[["acc", "f1", "coverage"]].mean().unstack("sequence_id")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
