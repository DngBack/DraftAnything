"""Calibration and selective prediction of the three-seed raft_v3 ensemble on OIAM M3/M4 (review P1; descriptive).

Per scored target: confidence = max ensemble probability (dustbin excluded, as for argmax), correct = argmax hits the
true ancestor. Reports expected calibration error (15 equal-mass bins), and accuracy when only the most confident
fraction of targets is kept (risk-coverage), plus per-window mean confidence vs accuracy (window-level flagging).

  python experiments/raft_calibration.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.genealogy_constrained_decode import CACHE, dense_p  # noqa: E402

COVER = (1.0, 0.9, 0.75, 0.5, 0.25)


def ece(conf, ok, bins=15):
    q = np.quantile(conf, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, conf, side="right") - 1, 0, bins - 1)
    return sum(abs(ok[idx == b].mean() - conf[idx == b].mean()) * np.mean(idx == b) for b in range(bins)
               if np.any(idx == b))


def main():
    with open(CACHE, "rb") as f:
        items = pickle.load(f)
    rows = []
    for w in items:
        p = dense_p(w)
        r, t = w["r"], w["truth"]
        rows.append(pd.DataFrame({"sequence": w["sequence"], "H": w["H"], "window_id": w["window_id"],
                                  "n_anchors": len(w["pa"]), "conf": p[r].max(1), "ok": p[r].argmax(1) == t}))
    d = pd.concat(rows, ignore_index=True)
    d["dense"] = d.n_anchors > 200
    d.to_parquet(RESULTS / "raft_calibration.parquet", index=False)
    out = []
    for (dense, h), g in [((False, h), g) for h, g in d.groupby("H")] + list(d[d.dense].groupby(["dense", "H"])):
        g = g.sort_values("conf", ascending=False)
        row = {"subset": "dense" if dense else "all", "H": h, "targets": len(g), "mean_conf": g.conf.mean(),
               "acc": g.ok.mean(), "ECE": ece(g.conf.to_numpy(), g.ok.to_numpy().astype(float))}
        for c in COVER[1:]:
            row[f"acc@{int(c * 100)}%"] = g.ok.iloc[:max(1, int(c * len(g)))].mean()
        out.append(row)
    print("target-weighted (paper reports window means):")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    wm = d.groupby(["H", "window_id"]).agg(conf=("conf", "mean"), acc=("ok", "mean"), n_anchors=("n_anchors", "first"))
    for h, g in wm.groupby("H"):
        rho = spearmanr(g.conf, g.acc)[0]
        print(f"H={h}: window mean confidence vs accuracy Spearman {rho:.2f} (n={len(g)}); "
              f"mean |conf - acc| {np.mean(np.abs(g.conf - g.acc)):.3f}")


if __name__ == "__main__":
    main()
