"""POST-HOC analysis of the blind external test (not part of the blind result).

The blind run (blind_external_test.py) failed H1/H2. Diagnosis: most blind fields
hold tens of separate microcolonies, while the model (and the radial baselines)
assume one colony expanding about its centroid. Here every window is split into
colonies before prediction: anchors are grouped by single-linkage at 3x the median
nearest-neighbour spacing, each target joins the colony of its nearest anchor, and
models and geometric baselines run per colony. Same windows, shuffles and models
as the blind run; the split rule was chosen once, from the diagnosis, and not tuned.

  python experiments/blind_percolony_posthoc.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ss
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.blind_external_test import SEED, movies, window  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402
from experiments.raft_external_eval import bbox_shape, plain  # noqa: E402

TAGS = ["raft_sim_loc", "raft_sim_loc_s1", "raft_sim_loc_s2", "raft_v1_loc"]
LINK = 3.0


def colonies(pa):
    if len(pa) < 2:
        return np.zeros(len(pa), int)
    sp = np.median(cKDTree(pa).query(pa, k=2)[0][:, 1])
    pairs = cKDTree(pa).query_pairs(LINK * sp, output_type="ndarray")
    adj = ss.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pa), len(pa)))
    return connected_components(adj, directed=False)[1]


def per_colony(e, models, minutes, shape):
    pa = e["anchor"][["center_y", "center_x"]].to_numpy(float)
    pb = e["target"][["center_y", "center_x"]].to_numpy(float)
    ca = colonies(pa)
    cb = ca[cKDTree(pa).query(pb)[1]]
    preds = {}
    for c in np.unique(cb):
        ia, ib = np.flatnonzero(ca == c), np.flatnonzero(cb == c)
        sub = {"anchor": e["anchor"].iloc[ia], "target": e["target"].iloc[ib],
               "cost": e["cost"][np.ix_(ib, ia)]}
        out = {f"col_{k}": v for k, v in geometry_assignments(sub).items()}
        fa, fb = plain(sub["anchor"]), plain(sub["target"])
        for tag, (model, margs) in models.items():
            if len(ia) < 2:
                out[f"col_{tag}"] = np.zeros(len(ib), int)
                continue
            prob, _, _ = predict(model, fa, fb, minutes, shape, no_shape=True, init=margs.get("init", "radial"))
            out[f"col_{tag}"] = prob[:, :-1].argmax(1)
        for k, v in out.items():
            preds.setdefault(k, np.zeros(len(pb), int))[ib] = ia[v]
    return preds, len(np.unique(ca))


def main():
    models = {t: load(t) for t in TAGS}
    rng = np.random.default_rng(SEED)
    rows = []
    for source, name, mpf, frames, keys, anc in movies():
        ts = sorted(frames)
        shape = bbox_shape(list(frames.values()))
        gaps = {m: int(round(m / mpf)) for m in (30, 60, 120) if round(m / mpf) < len(ts)}
        for minutes, g in gaps.items():
            for s in ts:
                if s + g not in frames:
                    continue
                e, valid, truth = window(frames, keys, s, s + g, lambda o, s=s, b=s + g: anc(b, o, s), rng)
                if not len(truth):
                    continue
                preds, n_col = per_colony(e, models, minutes, shape)
                for method, p in preds.items():
                    rows.append({"dataset": source, "sequence": name, "window_id": f"{s}_{s + g}",
                                 "gap_minutes": minutes, "n_anchors": len(e["anchor"]), "n_colonies": n_col,
                                 "method": method, "accuracy": float(np.mean(p[valid] == truth))})
        print(source, name, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "blind_percolony_posthoc.parquet", index=False)
    seq = out.groupby(["dataset", "sequence", "gap_minutes", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    print(seq.round(3).to_string())


if __name__ == "__main__":
    main()
