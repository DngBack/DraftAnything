"""Zero-shot genealogy-constrained decoding on external bacterial datasets (docs/genealogy_constrained_protocol.md).

Positions-only networks trained on RodSim alone (LineageRAFT raft_sim_loc*, direct matcher dm_sim_loc*; three seeds
each, decoded per seed and averaged per window, as in the paper's transfer table). No image is used: the pairwise
term uses each network's own same-family head, the growth prior uses the RodSim Fano factor. Nothing is fitted to a
real dataset. Same windows, detection shuffles (seed 20261002) and metrics as raft_external_eval.py.

  python experiments/genealogy_constrained_external.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.genealogy_constrained_decode import (EPS, fano_of, growth_prior, map_cardinality,  # noqa: E402
                                                       mean_field_gc)
from experiments.image_partition_decode import mean_field  # noqa: E402
from experiments.lineage_glue import spacing_of  # noqa: E402
from experiments.lineage_raft import predict  # noqa: E402
from experiments.partition_metrics import pair_f1  # noqa: E402
from experiments.raft_external_eval import Runner, plain, run  # noqa: E402

TAGS = {"raft": ("raft_sim_loc", "raft_sim_loc_s1", "raft_sim_loc_s2"),
        "dm": ("dm_sim_loc", "dm_sim_loc_s1", "dm_sim_loc_s2")}
CFG = json.loads((RESULTS / "growth_prior_sim.json").read_text())


def size_err(pred, truth, n_a):
    nt = np.bincount(truth, minlength=n_a)
    npd = np.bincount(pred, minlength=n_a)
    return float(np.abs(npd - nt)[nt > 0].mean())


class GCRunner(Runner):
    def window(self, dataset, seq, wid, minutes, e, valid, truth, shape):
        if not len(truth):
            return
        fa, fb = plain(e["anchor"]), plain(e["target"])
        n_a = len(fa["pos"])
        if n_a < 2:
            return
        pb = fb["pos"]
        cb = (pb - pb.mean(0)) / spacing_of(pb)
        w = {"pa": fa["pos"], "pb": pb, "cb": cb, "shape": shape}
        acc = {}
        for tag, (model, margs) in self.models.items():
            prob, (i, j, fam), centre = predict(model, fa, fb, minutes, shape, no_shape=True,
                                                init=margs.get("init", "radial"))
            p = prob[:, :-1] / np.maximum(prob[:, :-1].sum(1, keepdims=True), 1e-9)
            logp = np.log(np.maximum(p, 1e-9))
            fam = np.clip(fam, EPS, 1 - EPS)
            th = np.log(fam / (1 - fam))
            m, s2, _ = growth_prior(w, centre, float(prob[:, -1].sum()), fano_of(len(pb) / n_a, CFG))
            preds = {"argmax": p.argmax(1)}
            preds["potts"] = mean_field(logp, i, j, fam, 2.0) if len(i) else preds["argmax"]
            preds["gc_growth"] = map_cardinality(logp, 1.0, m, s2)[0]
            q, f = mean_field_gc(logp, i, j, th, 1.0, 1.0, m, s2) if len(i) else (None, 0.0)
            preds["gc_full"] = map_cardinality(logp + f, 1.0, m, s2)[0]
            family = next(k for k, v in TAGS.items() if tag in v)
            for name, pr in preds.items():
                g = pr[valid]
                acc.setdefault((family, name), []).append(
                    (np.mean(g == truth), pair_f1(truth, g)[2], size_err(g, truth, n_a)))
        for (family, name), v in acc.items():
            v = np.array(v, dtype=float)
            self.rows.append({"dataset": dataset, "sequence": seq, "window_id": wid, "gap_minutes": minutes,
                              "n_anchors": n_a, "n_scored": len(truth), "network": family, "decoder": name,
                              "accuracy": v[:, 0].mean(), "pair_f1": np.nanmean(v[:, 1]) if
                              np.isfinite(v[:, 1]).any() else np.nan, "size_err": v[:, 2].mean()})


def main():
    import experiments.raft_external_eval as rx
    rx.Runner = GCRunner
    args = type("A", (), {"tags": [t for v in TAGS.values() for t in v], "decoders": ["argmax"],
                          "ensemble": False, "datasets": ["FAST", "recA", "rpsM"], "out": "gc"})()
    rx.summary = summary
    run(args)


def summary(out):
    out.to_parquet(RESULTS / "genealogy_constrained_external.parquet", index=False)
    seq = out.groupby(["dataset", "network", "decoder", "sequence"])[["accuracy", "pair_f1", "size_err"]].mean()
    print(seq.groupby(["dataset", "network", "decoder"]).mean().round(4).to_string())


if __name__ == "__main__":
    main()
