"""Probe: choose between ancestry hypotheses per input with a label-free biological check.

Every live ancestor should have at least one descendant two snapshots later (no
death, no exit, in a growing colony). Nearest assignment violates this when
colonies expand (interior ancestors get no target); registration violates it when
cells barely move. Score each hypothesis by its coverage, the fraction of anchors
that receive at least one target, and keep the best; no label, no fitted weight.

Exploratory. The blind sequences were already used (blind_external_test.py), so
numbers on them are post hoc; OIAM M3/M4 are development data.

  python experiments/label_free_selection.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.blind_external_test import SEED, movies, window  # noqa: E402
from experiments.blind_percolony_posthoc import per_colony  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT  # noqa: E402
from experiments.lineage_glue import image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402
from experiments.raft_external_eval import bbox_shape, plain  # noqa: E402

TAG = "raft_sim_loc"


def hypotheses(e, models, minutes, shape):
    out = dict(geometry_assignments(e))
    fa, fb = plain(e["anchor"]), plain(e["target"])
    model, margs = models[TAG]
    if len(fa["pos"]) >= 2:
        prob, _, _ = predict(model, fa, fb, minutes, shape, no_shape=True, init=margs.get("init", "radial"))
        out["raft"] = prob[:, :-1].argmax(1)
    col, _ = per_colony(e, models, minutes, shape)
    out["raft_col"] = col[f"col_{TAG}"]
    return out


def score(p, na):
    sizes = np.bincount(p, minlength=na)
    return {"coverage": float(np.mean(sizes > 0)), "size_cv": float(sizes.std() / max(sizes.mean(), 1e-9))}


def record(rows, base, e, valid, truth, models, minutes, shape):
    na = len(e["anchor"])
    for name, p in hypotheses(e, models, minutes, shape).items():
        rows.append({**base, "method": name, "accuracy": float(np.mean(p[valid] == truth)), **score(p, na)})


def main():
    from cellanc.loader import Windows
    from experiments.family_set import examples
    models = {TAG: load(TAG)}
    rows = []
    windows = Windows(BENCH, ROOT / "data").windows
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        for h in (33, 66, 132):
            for e in examples(alias, windows, compensate=False, horizon=h):
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok):
                    continue
                record(rows, {"dataset": "OIAM", "sequence": alias, "window_id": e["window"]["window_id"],
                              "gap_minutes": h}, e, ok.target_detection_id.map(ro).to_numpy(),
                       ok.ancestor_detection_id.map(co).to_numpy(), models, h, shape)
        print("OIAM", alias, flush=True)
    rng = np.random.default_rng(SEED)
    for source, name, mpf, frames, keys, anc in movies():
        ts = sorted(frames)
        shape = bbox_shape(list(frames.values()))
        gaps = {m: int(round(m / mpf)) for m in (30, 60, 120) if round(m / mpf) < len(ts)}
        for minutes, g in gaps.items():
            for s in ts:
                if s + g not in frames:
                    continue
                e, valid, truth = window(frames, keys, s, s + g, lambda o, s=s, b=s + g: anc(b, o, s), rng)
                if len(truth):
                    record(rows, {"dataset": source, "sequence": name, "window_id": f"{s}_{s + g}",
                                  "gap_minutes": minutes}, e, valid, truth, models, minutes, shape)
        print(source, name, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(RESULTS / "label_free_selection.parquet", index=False)
    key = ["dataset", "sequence", "window_id", "gap_minutes"]
    for pool in (["nearest_anchor", "raft"], ["nearest_anchor", "raft_col"],
                 ["nearest_anchor", "centroid_translation", "raft", "raft_col"]):
        sub = d[d.method.isin(pool)]
        pick = sub.loc[sub.sort_values("size_cv").groupby(key).coverage.idxmax()]
        best = sub.groupby(key).accuracy.max().rename("oracle")
        t = pick.set_index(key)[["accuracy"]].join(best).reset_index()
        for m in pool:
            t = t.merge(sub[sub.method == m][key + ["accuracy"]].rename(columns={"accuracy": m}), on=key)
        seq = t.groupby(["dataset", "sequence", "gap_minutes"]).mean(numeric_only=True) \
            .groupby(["dataset", "sequence"]).mean()
        print("pool", pool, "\n", seq.round(3).to_string(), "\n", seq.groupby("dataset").mean().round(3).to_string())


if __name__ == "__main__":
    main()
