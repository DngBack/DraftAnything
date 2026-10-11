"""Calibration of single seeds vs the 3-seed ensemble on OIAM M3/M4 (windows of tab:oiam).

Tests the explanation that averaging seeds helps the genealogy decoder because the averaged
target x ancestor probabilities are better calibrated. Per scored target: confidence (max
probability), correctness of the argmax, and the probability of the true ancestor.
Reports ECE (15 equal-width bins), NLL of the true ancestor and Brier score of the top choice.

  python experiments/calibration.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_raft import BENCH, ROOT, frame_dict, image_shape_of, load, predict  # noqa: E402

SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
HS = (33, 66, 132)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return sum(abs(correct[k == b].mean() - conf[k == b].mean()) * np.mean(k == b) for b in range(bins) if np.any(k == b))


def main():
    from cellanc.loader import Windows
    from experiments.family_set import examples
    models = {t: load(t) for t in SEEDS}
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        for h in HS:
            for e in examples(alias, windows, compensate=False, horizon=h):
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok):
                    continue
                r = ok.target_detection_id.map(ro).to_numpy()
                truth = ok.ancestor_detection_id.map(co).to_numpy()
                probs = {}
                for t, (m, a) in models.items():
                    p = predict(m, frame_dict(e["anchor"]), frame_dict(e["target"]), h, shape,
                                a.get("no_shape", False), 1, init=a.get("init", "radial"))[0][:, :-1]
                    probs[t] = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
                probs["ens3"] = np.mean([probs[t] for t in SEEDS], 0)
                for name, p in probs.items():
                    q = p[r]
                    rows.append(pd.DataFrame({"H": h, "dense": len(co) > 200, "method": name,
                                              "conf": q.max(1), "correct": q.argmax(1) == truth,
                                              "p_true": q[np.arange(len(r)), truth]}))
            print(alias, h, flush=True)
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(RESULTS / "calibration.parquet", index=False)
    summary(d)


def summary(d=None):
    d = pd.read_parquet(RESULTS / "calibration.parquet") if d is None else d
    out = []
    for (h, name), g in d.groupby(["H", "method"]):
        out.append({"H": h, "method": name, "acc": g.correct.mean(), "conf": g.conf.mean(),
                    "ece": ece(g.conf.to_numpy(), g.correct.to_numpy().astype(float)),
                    "nll": -np.log(np.maximum(g.p_true, 1e-9)).mean(),
                    "brier": ((g.conf - g.correct) ** 2).mean()})
    print(pd.DataFrame(out).round(4).to_string(index=False))


if __name__ == "__main__":
    main() if sys.argv[1:] != ["summary"] else summary()
