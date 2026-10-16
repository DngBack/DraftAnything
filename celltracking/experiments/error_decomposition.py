"""Conditional error decomposition of LineageRAFT on OIAM M3/M4 (docs/error_decomposition_protocol.md).

Same network scores as genealogy_constrained_decode.py (three-seed raft_v3 ensemble, cached). For every scored
target the ensemble argmax error is attributed by order-free oracle interventions:

  retrieval     the true ancestor is not among the k_out=24 candidates of the final centre of any seed, so no
                decoder over these scores can recover it
  registration  fixed by giving the network the true family centre (centre_override = centroid of the true
                family in the target snapshot; network features unchanged, candidates and offsets re-scored)
  grouping      fixed by giving the decoder the true descendant partition (match_groups) on the network's scores
  both / either the 2x2 table of (fixed by centre) x (fixed by partition)
  residual      fixed by neither, nor by both together

  python experiments/error_decomposition.py run       # GPU (re-scoring), writes error_decomposition.parquet
  python experiments/error_decomposition.py summary
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.genealogy_constrained_decode import (  # noqa: E402
    CACHE, HS, SEEDS, dense_p, family_centres, featurized, raft_scores)
from experiments.lineage_glue import image_shape_of  # noqa: E402
from experiments.lineage_raft import load  # noqa: E402
from experiments.raft_family_probe import match_groups  # noqa: E402

OUT = RESULTS / "error_decomposition.parquet"
K_OUT = 24


def candidates(centre, cb, k=K_OUT):
    return cKDTree(centre).query(cb, k=min(k, len(centre)))[1].reshape(len(cb), -1)


def true_partition(la):
    return np.where(la >= 0, la, la.max() + 1 + np.arange(len(la)))


def run():
    with open(CACHE, "rb") as f:
        items = pickle.load(f)
    models = [load(t)[0] for t in SEEDS]
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        for h in HS:
            ex = {e["window"]["window_id"]: e for e in examples(alias, windows, compensate=False, horizon=h)}
            for w in (x for x in items if x["sequence"] == alias and x["H"] == h):
                n_a = len(w["pa"])
                r, t, la = w["r"], w["truth"], w["lab_all"]
                p = dense_p(w)
                logp = np.log(np.maximum(p, 1e-9))
                # oracle centre: true family centroid in the target snapshot (canonical units)
                cen = [family_centres(w["cb"][la >= 0], la[la >= 0], n_a, c) for c in w["centre"]]
                xa, xb, ca, cb = featurized(ex[w["window_id"]], h, shape)
                res = [raft_scores(m, xa, xb, ca, cb, torch.as_tensor(c, device="cuda", dtype=torch.float32))
                       for m, c in zip(models, cen)]
                po = dense_p(w, override=[x[0] for x in res])
                logpo = np.log(np.maximum(po, 1e-9))
                groups = true_partition(la)
                pred = {"argmax": p.argmax(1), "oracle_centre": po.argmax(1),
                        "oracle_partition": match_groups(logp, groups),
                        "oracle_both": match_groups(logpo, groups)}
                # candidate recall of the true ancestor (any seed / seed mean), network and oracle centre
                rec = np.stack([(candidates(c, w["cb"])[r] == t[:, None]).any(1) for c in w["centre"]])
                rec_o = np.stack([(candidates(c, w["cb"])[r] == t[:, None]).any(1) for c in cen])
                # centre error of the true ancestor's family, in median anchor spacings (canonical)
                sp = float(np.median(cKDTree(w["ca"]).query(w["ca"], k=2)[0][:, 1]))
                cerr = np.linalg.norm(np.mean(w["centre"], 0)[t] - cen[0][t], axis=1) / sp
                ok = {k: v[r] == t for k, v in pred.items()}
                for idx in range(len(r)):
                    rows.append({"sequence": alias, "H": h, "window_id": w["window_id"], "n_anchors": n_a,
                                 "in_cand_any": bool(rec[:, idx].any()), "in_cand_mean": float(rec[:, idx].mean()),
                                 "in_cand_oracle": bool(rec_o[:, idx].any()), "centre_err": float(cerr[idx]),
                                 **{f"ok_{k}": bool(v[idx]) for k, v in ok.items()}})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    return df


SIGMAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def sweep():
    """How precise must the family centre be? True centroids + isotropic noise of sigma anchor spacings
    (canonical units), re-scored; argmax and oracle-partition accuracy. H = 66, 132."""
    with open(CACHE, "rb") as f:
        items = pickle.load(f)
    models = [load(t)[0] for t in SEEDS]
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        for h in (66, 132):
            ex = {e["window"]["window_id"]: e for e in examples(alias, windows, compensate=False, horizon=h)}
            for w in (x for x in items if x["sequence"] == alias and x["H"] == h):
                n_a = len(w["pa"])
                r, t, la = w["r"], w["truth"], w["lab_all"]
                xa, xb, ca, cb = featurized(ex[w["window_id"]], h, shape)
                sp = float(np.median(cKDTree(w["ca"]).query(w["ca"], k=2)[0][:, 1]))
                has = np.bincount(la[la >= 0], minlength=n_a) > 0
                groups = true_partition(la)
                rng = np.random.default_rng(0)
                for sg in SIGMAS:
                    noise = rng.normal(0, sg * sp, (n_a, 2)) * has[:, None]
                    cen = [family_centres(w["cb"][la >= 0], la[la >= 0], n_a, c) + noise for c in w["centre"]]
                    res = [raft_scores(m, xa, xb, ca, cb, torch.as_tensor(c, device="cuda", dtype=torch.float32))
                           for m, c in zip(models, cen)]
                    po = dense_p(w, override=[x[0] for x in res])
                    rows.append({"sequence": alias, "H": h, "window_id": w["window_id"], "n_anchors": n_a,
                                 "sigma": sg, "acc_argmax": float(np.mean(po.argmax(1)[r] == t)),
                                 "acc_partition": float(np.mean(
                                     match_groups(np.log(np.maximum(po, 1e-9)), groups)[r] == t))})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "error_decomposition_sweep.parquet", index=False)
    df["dense"] = df.n_anchors > 200
    print(df.groupby(["dense", "H", "sigma"])[["acc_argmax", "acc_partition"]].mean().round(3).to_string())


def attribute(d):
    wrong = ~d.ok_argmax
    out = pd.Series(index=d.index, dtype=object)
    out[~wrong] = "correct"
    out[wrong & ~d.in_cand_any] = "retrieval"
    c, g, b = d.ok_oracle_centre, d.ok_oracle_partition, d.ok_oracle_both
    rest = wrong & d.in_cand_any
    out[rest & c & ~g] = "registration only"
    out[rest & ~c & g] = "grouping only"
    out[rest & c & g] = "either oracle"
    out[rest & ~c & ~g & b] = "needs both"
    out[rest & ~c & ~g & ~b] = "residual"
    return out


def boot_ci(d, col, n=2000, seed=0):
    """Window-cluster bootstrap of a window-mean statistic."""
    wm = d.groupby("window_id")[col].mean().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(wm), (n, len(wm)))
    s = wm[idx].mean(1)
    return wm.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def summary(df=None):
    df = pd.read_parquet(OUT) if df is None else df
    df["dense"] = df.n_anchors > 200
    df["cls"] = attribute(df)
    order = ["retrieval", "registration only", "either oracle", "grouping only", "needs both", "residual"]
    for subset, part in (("pooled", df), ("M4", df[df.sequence == "M4"]), ("M3", df[df.sequence == "M3"])):
        for dense in (False, True):
            s = part[part.dense] if dense else part
            if not len(s):
                continue
            print(f"\n== {subset}, {'dense' if dense else 'all'} windows (target-weighted) ==")
            acc = s.groupby("H")[[c for c in s if c.startswith("ok_")] +
                                 ["in_cand_any", "in_cand_mean", "in_cand_oracle"]].mean()
            print(acc.round(3).to_string())
            # window-mean accuracies (the paper's metric)
            wm = s.groupby(["H", "window_id"])[[c for c in s if c.startswith("ok_")]].mean().groupby("H").mean()
            print("window-mean:", wm.round(3).to_string(), sep="\n")
            err = s[s.cls != "correct"]
            share = err.groupby("H").cls.value_counts(normalize=True).unstack().reindex(columns=order).fillna(0)
            print("share of argmax errors:", share.round(3).to_string(), sep="\n")
            # accuracy conditional on retrieval
            print("acc | true ancestor in candidates:",
                  s[s.in_cand_any].groupby("H").ok_argmax.mean().round(3).to_dict())
            for h in HS:
                x = s[s.H == h]
                if x.window_id.nunique() >= 3:
                    print(f"  H={h} window CI: recall_any %.3f [%.3f, %.3f]" % boot_ci(x, "in_cand_any"),
                          "| oracle centre %.3f [%.3f, %.3f]" % boot_ci(x, "ok_oracle_centre"))


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        summary(run())
    elif sys.argv[1:] == ["sweep"]:
        sweep()
    else:
        summary()
