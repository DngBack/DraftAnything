"""Evaluate a trained embedding (train_embed.py) on benchmark windows, decoded by nearest
neighbour in learned-cost space, chained across the schedule exactly like baselines.chained_nearest.

Shape/density features are looked up via labels/gt_mapping (oracle, evaluator-side), the same
shortcut experiments/decide.py and pyuat_run.py use for their go/no-go features -- not something
a real detector-based pipeline could do yet (see docs v1 report, limits).

  uv run python experiments/eval_embed.py --tag v0 --seqs M3 --schedules ends s16 s32
    -> outputs/results/embed_h132_<tag>.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.embed import Embed, lap_decode, soft_transition  # noqa: E402
from cellanc.transport import log_plan_from_cost  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.train_embed import IDX, frame_features  # noqa: E402

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs"
BENCH = OUT / "benchmarks" / "v1"
SEQ_DIR = {f"M{i}": f"0{i}" for i in range(5)}
_state: dict = {}


def _load(alias: str):
    if alias not in _state:
        obs = pd.read_parquet(IDX / SEQ_DIR[alias] / "observations.parquet",
                              columns=["frame_index", "gt_track_id", "center_y", "center_x"])
        shp = pd.read_parquet(IDX / SEQ_DIR[alias] / "shape.parquet")
        gm = pd.read_parquet(BENCH / "labels/gt_mapping" / f"{alias}.parquet")
        m = gm.merge(obs, on=["frame_index", "gt_track_id"]).merge(
            shp, on=["frame_index", "gt_track_id"], how="left")
        tg = pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
        _state[alias] = (m, dict(list(tg.groupby("window_id"))))
    return _state[alias]


@torch.no_grad()
def embed_chain(w: dict, model, dev, m_by_frame: dict, time_step_min: float,
                decode: str = "nearest", cap: int | None = 2, temp: float = 1.0,
                eps: float = 1.0, tau: float | None = None, ot_hard: bool = False) -> dict[int, int]:
    frames = w["observed_frames"]
    e_cache = {}
    g_prev, t_prev = None, None
    for t in frames:
        g = m_by_frame[t]
        dt_prev = (t - t_prev) if t_prev is not None else None
        ids, feat = frame_features(g, g_prev, dt_prev)
        n_in = model.phi[0].in_features
        if feat.shape[1] != n_in:  # older checkpoint predating a later feature addition
            feat = feat[:, :n_in]
        local = dict(zip(g.gt_track_id, g.local_detection_id))
        e = model.embed(torch.as_tensor(feat, device=dev))
        e_cache[t] = (np.array([local[i] for i in ids]), e)
        g_prev, t_prev = g, t

    if decode == "dp":
        # Shortest-path DP over ALL frame pairs (not just consecutive), not just the immediate
        # predecessor: unlike a classical motion-likelihood tracker, this model's cost is defined
        # for any dt, so a direct anchor->frame_k comparison is available to cross-check the
        # chained route -- information per-hop LAP provably cannot use (see the separability
        # argument: independent per-hop capacitated LAP is already globally cost-optimal for a
        # chain-only objective, so any gain here can only come from this extra cross-frame signal).
        # No capacity constraint yet (isolating the skip-edge effect); compare against --decode
        # nearest (also uncapacitated) for a fair test.
        ids0 = e_cache[frames[0]][0]
        dp_cost = {frames[0]: np.zeros(len(ids0))}
        dp_anc = {frames[0]: np.arange(len(ids0))}
        for k, cur in enumerate(frames[1:], start=1):
            ids_c, e_c = e_cache[cur]
            best_cost = np.full(len(ids_c), np.inf)
            best_anc = np.zeros(len(ids_c), dtype=int)
            for prev in frames[:k]:
                ids_p, e_p = e_cache[prev]
                dt = (cur - prev) * time_step_min
                step = model.cost(e_p, e_c, dt).cpu().numpy()  # (n_prev, n_cur)
                total = dp_cost[prev][:, None] + step  # (n_prev, n_cur)
                j = total.argmin(0)
                cand_cost = total[j, np.arange(len(ids_c))]
                better = cand_cost < best_cost
                best_cost[better] = cand_cost[better]
                best_anc[better] = dp_anc[prev][j[better]]
            dp_cost[cur], dp_anc[cur] = best_cost, best_anc
        final_ids, _ = e_cache[frames[-1]]
        anc = ids0[dp_anc[frames[-1]]]
        return {int(c): int(a) for c, a in zip(final_ids, anc)}

    if decode == "ot":
        # Semi-relaxed entropic OT (cellanc/transport.py), reusing its mass-conservation/capacity
        # machinery -- built for exactly the "soft decode needs a real capacity constraint, not a
        # per-column softmax" gap the plain soft_transition decode above was missing -- but with
        # the learned embedding cost instead of transport.py's pixel-distance-after-expansion cost.
        # Mass = major_sd * minor_sd (area proxy), already in the shape features merged into m.
        ids0 = e_cache[frames[0]][0]
        mass = {t: (m_by_frame[t].major_sd.fillna(1.0) * m_by_frame[t].minor_sd.fillna(1.0))
                .to_numpy() for t in frames}
        A = np.eye(len(ids0))
        for prev, cur in zip(frames, frames[1:]):
            _, e_p = e_cache[prev]
            _, e_c = e_cache[cur]
            dt = (cur - prev) * time_step_min
            cost = model.cost(e_p, e_c, dt).cpu().numpy()
            lp = log_plan_from_cost(cost, mass[prev], mass[cur], eps, tau)
            cond = np.exp(lp - logsumexp(lp, axis=0, keepdims=True))  # P(parent | child)
            if ot_hard:
                cond = (cond == cond.max(0, keepdims=True)).astype(float)
                cond /= cond.sum(0, keepdims=True)
            A = A @ cond
        final_ids, _ = e_cache[frames[-1]]
        anc = ids0[A.argmax(0)]
        return {int(c): int(a) for c, a in zip(final_ids, anc)}

    if decode == "soft":
        ids0, e_prev = e_cache[frames[0]]
        belief = np.eye(len(ids0))  # (n0, n0): each src cell starts certain of itself
        for prev, cur in zip(frames, frames[1:]):
            ids_c, e_c = e_cache[cur]
            dt = (cur - prev) * time_step_min
            cost = model.cost(e_prev, e_c, dt).cpu().numpy()  # (n_prev, n_cur)
            trans = soft_transition(cost, temp)  # (n_prev, n_cur), column-stochastic
            belief = trans.T @ belief  # (n_cur, n0): mix of frame-0 beliefs per current cell
            e_prev = e_c
        final_ids, _ = e_cache[frames[-1]]
        anc = ids0[belief.argmax(1)]
        return {int(c): int(a) for c, a in zip(final_ids, anc)}

    lineage = {int(i): int(i) for i in e_cache[frames[0]][0]}
    for prev, cur in zip(frames, frames[1:]):
        ids_p, e_p = e_cache[prev]
        ids_c, e_c = e_cache[cur]
        dt = (cur - prev) * time_step_min
        cost = model.cost(e_p, e_c, dt).cpu().numpy()
        row = lap_decode(cost, cap) if decode == "lap" else cost.argmin(0)
        parent = ids_p[row]
        lineage = {int(c): lineage[int(p)] for c, p in zip(ids_c, parent)}
    return lineage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--seqs", nargs="+", default=["M3"])
    ap.add_argument("--H", type=int, default=132)
    ap.add_argument("--schedules", nargs="+", default=["ends", "s16", "s32"])
    ap.add_argument("--decode", choices=["nearest", "lap", "soft", "dp", "ot"], default="nearest")
    ap.add_argument("--cap", default="2", help="int, or 'soft' for the family-ratio cap")
    ap.add_argument("--temp", type=float, default=1.0, help="softmax temperature for --decode soft")
    ap.add_argument("--eps", type=float, default=1.0, help="Sinkhorn entropy reg for --decode ot")
    ap.add_argument("--tau", type=float, default=None, help="Sinkhorn KL capacity price for "
                    "--decode ot; omit for balanced OT")
    ap.add_argument("--ot-hard", action="store_true", help="commit to argmax parent at every "
                    "link for --decode ot, instead of multiplying soft conditionals through")
    args = ap.parse_args()
    args.cap = None if args.cap == "soft" else int(args.cap)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ROOT / "outputs/results" / f"embed_{args.tag}.pt", map_location=dev)
    model = Embed(n_in=state["phi.0.weight"].shape[1]).to(dev)
    model.load_state_dict(state)
    model.eval()

    windows = Windows(BENCH, ROOT / "data")
    ws = [w for w in windows.windows if w["sequence_id"] in args.seqs
         and w["horizon_frames"] == args.H and w["schedule_id"] in args.schedules
         and f"A_h{args.H}" in w["experiments"]]
    print(len(ws), "windows", flush=True)

    rows = []
    for w in ws:
        m, tg_by = _load(w["sequence_id"])
        m_by_frame = dict(list(m[m.frame_index.isin(w["observed_frames"])].groupby("frame_index")))
        time_step_min = w["horizon_min"] / w["horizon_frames"]  # minutes per frame, any schedule
        pred = embed_chain(w, model, dev, m_by_frame, time_step_min, args.decode, args.cap, args.temp,
                          args.eps, args.tau, args.ot_hard)
        s = score_window(pred, set(), tg_by[w["window_id"]], pd.DataFrame(columns=["det_1", "det_2"]))
        cap_tag = ("soft" if args.cap is None else str(args.cap)) if args.decode == "lap" else ""
        tau_str = "inf" if args.tau is None else f"{args.tau:g}"
        ot_tag = f"_ot_e{args.eps:g}_t{tau_str}" if args.decode == "ot" else ""
        if args.decode == "ot" and args.ot_hard:
            ot_tag += "_hard"
        suffix = (f"_lap{cap_tag}" if cap_tag else
                 f"_soft_t{args.temp:g}" if args.decode == "soft" else
                 "_dp" if args.decode == "dp" else ot_tag)
        method = f"embed_{args.tag}" + suffix
        rows.append({"method": method, "window_id": w["window_id"],
                    "sequence_id": w["sequence_id"], "schedule_id": w["schedule_id"],
                    "horizon_frames": w["horizon_frames"],
                    **{k: s[k] for k in ["n_scored", "n_correct", "set_f1_sum", "count_n_anchors"]}})
    R = pd.DataFrame(rows)
    out = OUT / "results" / f"embed_h{args.H}_{args.tag}{suffix}.parquet"
    R.to_parquet(out, index=False)
    R["acc"] = R.n_correct / R.n_scored
    print(R.groupby(["schedule_id", "sequence_id"]).acc.mean().round(3))
    print("saved", out)


if __name__ == "__main__":
    main()
