"""Bio-informed decoding: a binary-fission prior on family sizes, solved exactly.

Argmax treats every target independently. Binary fission makes family sizes
under-dispersed: after H minutes an ancestor has about m = N_B / N_A descendants,
in powers of two blurred by cell-cycle asynchrony, and almost never none. The
decoder maximises  sum_t log q(t, a_t) + lam * sum_a log pi(n_a)  with
  pi(n) ~ exp(-(log2 n - log2 m)^2 / (2 s^2))  for n >= 1,   pi(0) = eps,
s = 0.5 generations, eps = 0.05 (fixed a priori, not tuned). Each ancestor gets K
slots whose costs are the slopes of the lower convex hull of -log pi (non-decreasing,
so slots fill in order); targets are matched to slots of their top-C candidate
ancestors by an exact sparse min-cost bipartite matching.

  python experiments/fission_prior_decode.py --tags raft_v3 raft_sim_loc
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ss
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import BENCH, RESULTS, ROOT  # noqa: E402
from experiments.lineage_glue import frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402


def slot_costs(m, s=0.5, eps=0.05, K=None):
    K = K or int(math.ceil(4 * m)) + 3
    n = np.arange(1, K + 1)
    logpi = np.concatenate([[math.log(eps)], -(np.log2(n) - math.log2(max(m, 1e-6))) ** 2 / (2 * s * s)])
    logpi[1:] += math.log(1 - eps) - np.log(np.exp(logpi[1:]).sum())
    # slopes of the lower convex hull of F(n) = -log pi(n): non-decreasing slot costs
    F = -logpi
    hull = [0]
    for k in range(1, len(F)):
        while len(hull) >= 2 and (F[hull[-1]] - F[hull[-2]]) * (k - hull[-1]) >= (F[k] - F[hull[-1]]) * (hull[-1] - hull[-2]):
            hull.pop()
        hull.append(k)
    env = np.interp(np.arange(len(F)), hull, F[hull])
    return np.diff(env)


def fission_decode(q, n_anchor, lam=1.0, s=0.5, eps=0.05, C=8):
    nb = len(q)
    m = nb / max(n_anchor, 1)
    cost_k = lam * slot_costs(m, s, eps)
    K = len(cost_k)
    C = min(C, n_anchor)
    cand = np.argsort(-q, 1)[:, :C]
    lq = -np.log(np.maximum(np.take_along_axis(q, cand, 1), 1e-12))
    rows = np.repeat(np.arange(nb), C * K)
    cols = (cand[:, :, None] * K + np.arange(K)[None, None]).reshape(-1)
    w = (lq[:, :, None] + cost_k[None, None]).reshape(-1)
    w = w - w.min() + 1.0                                   # positive; a constant per matched row
    g = ss.csr_matrix((w, (rows, cols)), shape=(nb, n_anchor * K))
    try:
        _, col = min_weight_full_bipartite_matching(g)
    except ValueError:
        return q.argmax(1)
    return col // K


def main():
    from cellanc.loader import Windows
    from experiments.family_set import examples
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["raft_v3", "raft_sim_loc"])
    ap.add_argument("--lams", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--out", default="fission_prior_decode")
    a = ap.parse_args()
    models = {t: load(t) for t in a.tags}
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        for h in (33, 66, 132):
            for e in examples(alias, windows, compensate=False, horizon=h):
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                anc, tgt = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
                ro = {int(v): i for i, v in enumerate(tgt.local_detection_id)}
                co = {int(v): i for i, v in enumerate(anc.local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok) or len(anc) < 2:
                    continue
                r, truth = ok.target_detection_id.map(ro).to_numpy(), ok.ancestor_detection_id.map(co).to_numpy()
                for tag, (model, margs) in models.items():
                    prob, _, _ = predict(model, frame_dict(anc), frame_dict(tgt), h, shape,
                                         margs.get("no_shape", False), init=margs.get("init", "radial"))
                    q = prob[:, :-1] / np.maximum(prob[:, :-1].sum(1, keepdims=True), 1e-9)
                    preds = {"argmax": q.argmax(1)}
                    for lam in a.lams:
                        preds[f"fission{lam:g}"] = fission_decode(q, len(anc), lam)
                    for name, p in preds.items():
                        rows.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"],
                                     "n_anchors": len(anc), "model": tag, "decoder": name,
                                     "accuracy": float(np.mean(p[r] == truth))})
            print(alias, h, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(RESULTS / f"{a.out}.parquet", index=False)
    print(d.pivot_table(index=["model", "decoder"], columns="H", values="accuracy").round(3).to_string())
    print("dense\n", d[d.n_anchors > 200].pivot_table(index=["model", "decoder"], columns="H",
                                                      values="accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
