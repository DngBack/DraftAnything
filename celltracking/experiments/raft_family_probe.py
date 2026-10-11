"""Where is the family signal lost? Probe a trained LineageRAFT on OIAM M3/M4.

Reports per horizon: argmax accuracy, oracle-partition accuracy using the
model's own probabilities (ceiling of any family decoder on these scores),
same-family AP of the family head and of the probability overlap p_i.p_k on
target kNN pairs, and an average-linkage cluster decoder with one-to-one
cluster->anchor matching. Development analysis; partition labels are GT.

  python experiments/raft_family_probe.py --tag raft_v1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.lineage_glue import frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402


def ap(score, y):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-score)
    y = y[o]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def match_groups(logp, groups):
    """One anchor per group, anchors reused only after all are used."""
    labels = np.unique(groups)
    block = -np.stack([logp[groups == g].sum(0) for g in labels])
    n_g, n_a = block.shape
    reps = int(np.ceil(n_g / n_a))
    big = np.abs(block).max() + 1
    rows, cols = linear_sum_assignment(np.concatenate([block + k * big for k in range(reps)], 1))
    out = np.empty(len(groups), int)
    for r, c in zip(rows, cols):
        out[groups == labels[r]] = c % n_a
    return out


def average_linkage(n, i, j, w, tau):
    """Average-linkage on a sparse graph (missing edges count as 0); stop below tau."""
    import heapq
    adj = [dict() for _ in range(n)]
    for x, y, v in zip(i, j, w):
        if x != y:
            adj[x][y] = adj[x].get(y, 0) + v / 2
            adj[y][x] = adj[y].get(x, 0) + v / 2
    size = np.ones(n)
    ver = np.zeros(n, int)
    alive = np.ones(n, bool)
    heap = [(-v / (size[x] * size[y]), x, y, 0, 0) for x in range(n) for y, v in adj[x].items() if x < y]
    heapq.heapify(heap)
    lab = np.arange(n)
    members = {x: [x] for x in range(n)}
    while heap:
        negv, x, y, vx, vy = heapq.heappop(heap)
        if -negv < tau:
            break
        if not (alive[x] and alive[y]) or ver[x] != vx or ver[y] != vy:
            continue
        for c, v in adj[y].items():
            if c != x:
                adj[x][c] = adj[x].get(c, 0) + v
                adj[c][x] = adj[c].get(x, 0) + v
            adj[c].pop(y, None)
        adj[x].pop(y, None)
        alive[y] = False
        size[x] += size[y]
        members[x] += members.pop(y)
        ver[x] += 1
        for c, v in adj[x].items():
            heapq.heappush(heap, (-v / (size[x] * size[c]), min(x, c), max(x, c),
                                  ver[min(x, c)], ver[max(x, c)]))
    for root, ms in members.items():
        lab[ms] = root
    return lab


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tag", default="raft_v1")
    ap_.add_argument("--max-dense", type=int, default=400)
    a = ap_.parse_args()
    model, margs = load(a.tag)
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        for h in (33, 66, 132):
            for e in examples(alias, windows, compensate=False, horizon=h):
                prob, (i, j, wf), _ = predict(model, frame_dict(e["anchor"]), frame_dict(e["target"]),
                                              h, shape, margs.get("no_shape", False),
                                              init=margs.get("init", "radial"))
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): k for k, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): k for k, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok) or len(co) < 2:
                    continue
                r = ok.target_detection_id.map(ro).to_numpy()
                truth = ok.ancestor_detection_id.map(co).to_numpy()
                p = prob[:, :-1] / np.maximum(prob[:, :-1].sum(1, keepdims=True), 1e-9)
                logp = np.log(np.maximum(p, 1e-9))
                lab = -np.ones(len(p), int)
                lab[r] = truth
                row = {"sequence": alias, "H": h, "n_anchors": len(co),
                       "argmax": float(np.mean(p[r].argmax(1) == truth)),
                       "oracle_partition": float(np.mean(match_groups(logp[r], truth) == truth))}
                if len(i):
                    keep = (lab[i] >= 0) & (lab[j] >= 0)
                    y = (lab[i] == lab[j])[keep]
                    w_self = (p[i] * p[j]).sum(1)
                    row["ap_family_head"] = ap(wf[keep], y)
                    row["ap_prob_overlap"] = ap(w_self[keep], y)
                    row["edge_pos_rate"] = float(y.mean())
                    if len(co) <= a.max_dense:
                        for tau in (0.3, 0.5):
                            g = average_linkage(len(p), i, j, wf, tau)
                            row[f"cluster_fam{tau}"] = float(np.mean(match_groups(logp, g)[r] == truth))
                            g = average_linkage(len(p), i, j, w_self, tau)
                            row[f"cluster_self{tau}"] = float(np.mean(match_groups(logp, g)[r] == truth))
                rows.append(row)
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"raft_family_probe_{a.tag}.parquet", index=False)
    df["dense"] = df.n_anchors > 200
    cols = [c for c in df.columns if c not in ("sequence", "H", "n_anchors", "dense")]
    print(df.groupby("H")[cols].mean().round(4).T.to_string())
    print("dense\n", df[df.dense].groupby("H")[cols].mean().round(4).T.to_string())


if __name__ == "__main__":
    main()
