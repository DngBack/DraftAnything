"""Growth-heterogeneity readout from predicted ancestry (docs/offspring_readout_protocol.md).

Offspring number per ancestor over the gap, from the existing prediction caches of OIAM M3/M4 decoder windows:
S1 = relative error of the coefficient of variation of offspring numbers, S2 = Spearman correlation of
predicted vs true offspring numbers. CPU only, run once.

  python experiments/offspring_readout.py
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.decoder_on_baselines import CACHE as TRA_CACHE, FLOOR, fit_t, log_softmax, paired_ci  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402
from experiments.image_partition_decode import mean_field  # noqa: E402

HS = (33, 66, 132)
SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")


def cv(n):
    n = n[n > 0]
    return float(n.std() / n.mean()) if len(n) else np.nan


def stats(pred, truth, n_anchor):
    nt = np.bincount(truth, minlength=n_anchor)
    npr = np.bincount(pred, minlength=n_anchor)
    c_t, c_p = cv(nt), cv(npr)
    s1 = abs(c_p - c_t) / c_t if c_t > 0 else np.nan
    u = (nt > 0) | (npr > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s2 = spearmanr(npr[u], nt[u])[0] if len(np.unique(nt[u])) > 1 and len(np.unique(npr[u])) > 1 else np.nan
    return {"cv_true": c_t, "cv_pred": c_p, "s1": s1, "s2": s2}


def main():
    with open(RESULTS / "partition_metrics_cache.pkl", "rb") as f:
        raft = {(it["H"], it["window_id"]): it for it in pickle.load(f)}
    with open(TRA_CACHE, "rb") as f:
        tra = pickle.load(f)
    temps = {h: fit_t(tra, h) for h in HS}
    rows = []
    for it in tra:
        if it["sequence"] == "M2" or (it["H"], it["window_id"]) not in raft:
            continue
        ri = raft[(it["H"], it["window_id"])]
        r, truth, h = it["r"], it["truth"], it["H"]
        assert np.array_equal(ri["truth"], truth)
        na = it["p_tra"].shape[1]
        lp = np.log(np.maximum(it["p_tra"].astype(np.float64), FLOOR))
        preds = {"nearest": ri["pred"]["nearest"], "gbdt_own": it["gbdt_own"][r],
                 "trackastra_argmax": lp.argmax(1)[r],
                 "trackastra_mf2": mean_field(log_softmax(lp / temps[h]), *it["edges"], 2.0)[r],
                 "trackastra_mf2_T1": mean_field(lp, *it["edges"], 2.0)[r],
                 "ens_mf2": ri["pred"]["ens_mf2"], "ens_noann_mf2": ri["pred"]["ens_noann_mf2"]}
        per_seed = {f"raft_{d}": [stats(np.asarray(ri["pred"][f"{t}_{d}"]), truth, na) for t in SEEDS]
                    for d in ("argmax", "mf2")}
        for name, p in preds.items():
            rows.append({"H": h, "window_id": it["window_id"], "n_anchors": it["n_anchors"], "method": name,
                         **stats(np.asarray(p), truth, na)})
        for name, ss in per_seed.items():
            rows.append({"H": h, "window_id": it["window_id"], "n_anchors": it["n_anchors"], "method": name,
                         **{k: float(np.nanmean([s[k] for s in ss])) if not all(np.isnan(s[k]) for s in ss) else np.nan
                            for k in ss[0]}})
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "offspring_readout.parquet", index=False)
    pd.set_option("display.width", 250)
    for lab, sub in (("all", df), ("dense", df[df.n_anchors > 200])):
        print(f"== {lab}: mean over windows")
        print(sub.groupby(["method", "H"])[["cv_true", "cv_pred", "s1", "s2"]].mean().unstack("H").round(3).to_string())
    out = {}
    for stat, better in (("s1", -1), ("s2", 1)):
        w = df.pivot_table(index=["H", "window_id"], columns="method", values=stat)
        for a, b in (("ens_mf2", "trackastra_mf2"), ("ens_mf2", "nearest"), ("ens_noann_mf2", "trackastra_mf2"),
                     ("ens_noann_mf2", "nearest"), ("ens_mf2", "trackastra_mf2_T1"), ("raft_mf2", "trackastra_mf2")):
            for h in HS:
                x = w.loc[h][[a, b]].dropna()
                m, lo, hi = paired_ci(x[a], x[b])
                out[(stat, a, b, h)] = (m, lo, hi)
                print(f"{stat} {a} - {b} H={h} n={len(x)}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    b1 = all(out[("s1", "ens_mf2", b, h)][2] < 0 for b in ("trackastra_mf2", "nearest") for h in (66, 132))
    b2 = all(out[("s2", "ens_mf2", "trackastra_mf2", h)][0] > 0 for h in HS) and out[("s2", "ens_mf2", "trackastra_mf2", 66)][1] > 0
    b3 = all(out[("s1", "ens_noann_mf2", b, h)][2] < 0 for b in ("trackastra_mf2", "nearest") for h in (66, 132))
    print(f"fitted T: {temps}\nB1 holds: {b1}\nB2 holds: {b2}\nB3 holds: {b3}")


if __name__ == "__main__":
    main()
