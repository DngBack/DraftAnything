#!/usr/bin/env python3
"""Key numbers for the locked PHOEBI test (results/test/phoebi_v1): raw pooled and per-partition
rank-normalised mean AUROC, macro AP, paired deltas vs references; unit = partition:combination."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from popev import metrics
from popev.experiment import load_preds
K = ["bs", "bt", "fj", "ka", "mx", "pf"]; D = ROOT / "results/test/phoebi_v1"
M = ["mean_lin", "meanstd_lin", "deepsets", "abmil", "abmil_cla1", "tileonly_q50", "tileonly_mean"]
for n in (1, 4):
    print(f"== n_fields = {n}", flush=True)
    L = {}
    for m in M:
        S, P, Y, C, _ = load_preds(D, m, n); G = np.load(D / f"preds_{m}_n{n}.npz")["G"]
        R = metrics.rank_within(S, G); L[m] = (S, P, R, Y, C)
        raw = metrics.report(S, Y, P, C, K, n_boot=500); rk = metrics.report(R, Y, P, C, K, n_boot=500)
        print(f"{m:14s} raw AUROC {metrics.fmt(raw['auroc_mean'])} | ranked AUROC {metrics.fmt(rk['auroc_mean'])} mAP {rk['macro_ap']['point']:.3f} | "
              + " ".join(f"{k}:{rk['per_species'][k]['auroc']:.2f}" for k in K), flush=True)
    for a, b in (("tileonly_mean", "meanstd_lin"), ("tileonly_mean", "mean_lin"), ("tileonly_mean", "abmil"),
                 ("tileonly_q50", "meanstd_lin"), ("abmil_cla1", "abmil")):
        Sa, Pa, Ra, Y, C = L[a]; Sb, Pb, Rb, _, _ = L[b]
        d = metrics.paired_delta((Ra, Pa), (Rb, Pb), Y, C, n_boot=1000)
        print(f"  {a} - {b}: ranked dAUROC {metrics.fmt(d['auroc_mean'])} P>0 {d['auroc_mean']['p_gt0']:.3f} | dAP {metrics.fmt(d['macro_ap'])}", flush=True)
