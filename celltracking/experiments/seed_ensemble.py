"""Seed ensemble of the full model on OIAM M3/M4 (the windows of tab:oiam).

Fixed before running: the main candidate is `ens3`, the mean of the target x ancestor
probabilities of the three v3 training seeds, decoded by argmax (no tuning).
`ens3_darcy` adds the 30k-step Darcy-initialised model, which is the best single model at
132 minutes; it was chosen after seeing M3/M4, so it is exploratory and post hoc.
The single models are re-scored as a check against raft_eval_*.

  python experiments/seed_ensemble.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_raft import BENCH, ROOT, frame_dict, image_shape_of, load, predict  # noqa: E402

SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
EXTRA = "raft_v2_darcy_30k"
HS = (33, 66, 132)


def main():
    from cellanc.loader import Windows
    from experiments.family_set import examples
    models = {t: load(t) for t in (*SEEDS, EXTRA)}
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
                preds = {t: p.argmax(1) for t, p in probs.items()}
                preds["ens3"] = np.mean([probs[t] for t in SEEDS], 0).argmax(1)
                preds["ens3_darcy"] = np.mean([probs[t] for t in (*SEEDS, EXTRA)], 0).argmax(1)
                for name, pred in preds.items():
                    rows.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"],
                                 "n_anchors": len(co), "n_scored": len(r), "method": name,
                                 "accuracy": float(np.mean(pred[r] == truth))})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "seed_ensemble.parquet", index=False)
    df = df.assign(dense=df.n_anchors > 200)
    print(df.groupby(["method", "H"]).accuracy.mean().unstack().round(4).to_string())
    print("dense\n", df[df.dense].groupby(["method", "H"]).accuracy.mean().unstack().round(4).to_string())


if __name__ == "__main__":
    main()
