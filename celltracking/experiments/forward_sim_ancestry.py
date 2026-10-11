"""Forward-simulation ancestry (FSA): analysis by synthesis, no training, no labels.

Snapshot A is turned into a RodSim colony (positions, orientations, lengths from the
masks' second moments; pixel scale from cell width) and simulated forward for H
minutes, R times, with the unknown physics drawn from the simulator prior and the
doubling time read off the data, H ln2 / ln(N_B / N_A). Every simulated descendant
remembers its ancestor in A. Each rollout is aligned to snapshot B by a similarity
transform (centroid and RMS radius), and every real target is assigned to the
anchor whose simulated descendants have the highest kernel density at its position.

  python experiments/forward_sim_ancestry.py --rollouts 8
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.colony_sim import Colony, random_params  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT  # noqa: E402

SQ12 = math.sqrt(12)


def rollout(anchor, n_b, minutes, seed, doubling=None):
    """Simulate snapshot A forward; return descendant positions (px, y/x) and anchor index."""
    rng = np.random.default_rng(seed)
    p = random_params(rng)
    px_um = np.median(anchor.minor_sd) / 0.25                      # sim: minor_um = 0.25 for width 1
    length = np.maximum(anchor.major_sd.to_numpy() * SQ12 / px_um, 1.05)
    growth = max(n_b / len(anchor), 1.0001)
    p["doubling"] = doubling or float(np.clip(minutes * math.log(2) / math.log(growth), 15, 2000))
    p["birth_len"] = float(np.median(length) / 1.45)
    p["n_init"] = len(anchor)
    col = Colony(p, rng)
    n = len(anchor)
    col.pos = anchor[["center_x", "center_y"]].to_numpy(float) / px_um          # sim works in (x, y)
    col.theta = np.arctan2(anchor.axis_y.to_numpy(), anchor.axis_x.to_numpy())
    col.length = length
    col.born_len = length / rng.uniform(1.0, 1.9, n)
    col.width = np.ones(n)
    label = {int(t): k for k, t in enumerate(col.track)}
    for minute in range(int(round(minutes))):
        col.step(minute + 1)
    anc = []
    for t in col.track:
        t = int(t)
        while t not in label:
            t = col.parent[t]
        anc.append(label[t])
    return col.pos[:, ::-1] * px_um, np.array(anc)


def warp(pos, pb, sigma, iters=15, width=4.0, step=0.5):
    """Smooth non-rigid pull of the simulated template onto the observed targets:
    each simulated cell moves towards its nearest observed cell (and each observed
    cell pulls its nearest simulated cell), displacements are Gaussian-smoothed over
    `width` spacings so whole neighbourhoods move together, never single cells."""
    tb = cKDTree(pb)
    for _ in range(iters):
        ts = cKDTree(pos)
        disp = np.zeros_like(pos)
        cnt = np.zeros(len(pos))
        _, j = tb.query(pos)
        disp += pb[j] - pos
        cnt += 1
        _, i = ts.query(pb)
        np.add.at(disp, i, pb - pos[i])
        np.add.at(cnt, i, 1)
        disp /= cnt[:, None]
        pairs = ts.query_pairs(width * sigma, output_type="ndarray")
        w = np.exp(-((pos[pairs[:, 0]] - pos[pairs[:, 1]]) ** 2).sum(1) / (2 * (width * sigma / 2) ** 2))
        num, den = disp.copy(), np.ones(len(pos))
        np.add.at(num, pairs[:, 0], w[:, None] * disp[pairs[:, 1]])
        np.add.at(num, pairs[:, 1], w[:, None] * disp[pairs[:, 0]])
        np.add.at(den, pairs[:, 0], w)
        np.add.at(den, pairs[:, 1], w)
        pos = pos + step * num / den[:, None]
    return pos


def assign(anchor, target, sims, nonrigid=False, align="similarity"):
    pb = target[["center_y", "center_x"]].to_numpy(float)
    mu_b = pb.mean(0)
    rms_b = np.sqrt(((pb - mu_b) ** 2).sum(1).mean())
    sigma = np.median(cKDTree(pb).query(pb, k=2)[0][:, 1])
    score = np.zeros((len(pb), len(anchor)))
    for pos, anc in sims:
        mu_s = pos.mean(0)
        rms_s = np.sqrt(((pos - mu_s) ** 2).sum(1).mean())
        if align == "similarity":
            pos = mu_b + (pos - mu_s) * (rms_b / max(rms_s, 1e-9))
        if nonrigid:
            pos = warp(pos, pb, sigma)
        tree = cKDTree(pos)
        for i, nb in enumerate(tree.query_ball_point(pb, 3 * sigma)):
            if nb:
                d2 = ((pos[nb] - pb[i]) ** 2).sum(1)
                np.add.at(score[i], anc[nb], np.exp(-d2 / (2 * sigma ** 2)))
            else:
                score[i, anc[tree.query(pb[i])[1]]] += 1e-9
    return score.argmax(1), score


def calibrate_doubling(anc, n_b, h, fov, iters=3):
    """Doubling time such that the simulated cells *inside the field of view* match the
    observed count (cells leave the image once the colony outgrows it)."""
    rate = math.log(max(n_b / len(anc), 1.0001)) / h
    for k in range(iters):
        pos, _ = rollout(anc, n_b, h, 1000 + k, doubling=math.log(2) / rate)
        inside = np.sum((pos[:, 0] >= 0) & (pos[:, 0] < fov[0]) & (pos[:, 1] >= 0) & (pos[:, 1] < fov[1]))
        rate = max(rate + math.log(n_b / max(inside, 1)) / h, 1e-4)
    return float(np.clip(math.log(2) / rate, 15, 2000))


def run_window(e, h, rollouts, nonrigid=False, align="similarity", keep=0, fov=None):
    tg = e["targets"]
    ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
    anc, tgt = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
    ro = {int(v): i for i, v in enumerate(tgt.local_detection_id)}
    co = {int(v): i for i, v in enumerate(anc.local_detection_id)}
    ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
    if not len(ok) or len(anc) < 2:
        return []
    r, truth = ok.target_detection_id.map(ro).to_numpy(), ok.ancestor_detection_id.map(co).to_numpy()
    w = e["window"]
    dbl = calibrate_doubling(anc, len(tgt), h, fov) if fov is not None else None
    sims = [rollout(anc, len(tgt), h, k, doubling=dbl) for k in range(rollouts)]
    if keep:  # ABC rejection: keep the rollouts whose final layout is closest to snapshot B
        pb_ = tgt[["center_y", "center_x"]].to_numpy(float)
        tb = cKDTree(pb_)
        chamfer = [tb.query(pos)[0].mean() + cKDTree(pos).query(pb_)[0].mean() for pos, _ in sims]
        sims = [sims[k] for k in np.argsort(chamfer)[:keep]]
    pred, _ = assign(anc, tgt, sims, nonrigid, align)
    pa, pb = anc[["center_y", "center_x"]].to_numpy(float), tgt[["center_y", "center_x"]].to_numpy(float)
    near = cKDTree(pa).query(pb)[1]
    base = {"sequence": w["sequence_id"], "H": h, "window_id": w["window_id"], "n_anchors": len(anc)}
    return [{**base, "method": m, "accuracy": float(np.mean(p[r] == truth))}
            for m, p in (("fsa", pred), ("nearest", near))]


def main():
    from cellanc.loader import Windows
    from experiments.family_set import examples
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--max-anchors", type=int, default=100000)
    ap.add_argument("--out", default="forward_sim_ancestry")
    ap.add_argument("--nonrigid", action="store_true")
    ap.add_argument("--align", default="similarity", choices=["similarity", "none"])
    ap.add_argument("--min-anchors", type=int, default=0)
    ap.add_argument("--keep", type=int, default=0, help="ABC: keep the best K rollouts (0 = all)")
    ap.add_argument("--fov-calib", action="store_true", help="calibrate doubling on in-view cell counts")
    a = ap.parse_args()
    windows = Windows(BENCH, ROOT / "data").windows
    from experiments.lineage_glue import image_shape_of
    tasks = [(e, h, image_shape_of(alias) if a.fov_calib else None) for alias in ("M3", "M4") for h in (33, 66, 132)
             for e in examples(alias, windows, compensate=False, horizon=h) if a.min_anchors <= len(e["anchor"]) <= a.max_anchors]
    tasks.sort(key=lambda t: -len(t[0]["anchor"]) * t[1])
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_window)(e, h, a.rollouts, a.nonrigid, a.align, a.keep, fov)
                                              for e, h, fov in tasks)
    d = pd.DataFrame([x for r in rows for x in r])
    d.to_parquet(RESULTS / f"{a.out}.parquet", index=False)
    print(d.pivot_table(index="method", columns="H", values="accuracy").round(3).to_string())
    print("dense\n", d[d.n_anchors > 200].pivot_table(index="method", columns="H", values="accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
