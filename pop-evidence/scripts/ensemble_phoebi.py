#!/usr/bin/env python3
"""Equal-weight rank ensembles on PHOEBI dev (no tuned weights): per fold block and species, each
member's seed-averaged scores are converted to ranks within the evaluation bags (unsupervised), then
averaged. Usage: ensemble_phoebi.py N ref_run:ref_method member_run:member [member ...]"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import rankdata
from popev import metrics
from popev.experiment import load_preds
from popev.phoebi import Manifest
K = ["bs", "bt", "fj", "ka", "mx", "pf"]; n = int(sys.argv[1])
man = Manifest.default(); fold_of = {c: f for f, fd in enumerate(man.folds) for c in fd["dev_combos"]}
def ranks(S, C):
    S = S.mean(0); R = np.zeros_like(S); F = np.array([fold_of[c] for c in C])
    for f in np.unique(F):
        ii = F == f
        for k in range(S.shape[1]):
            R[ii, k] = rankdata(S[ii, k]) / ii.sum()
    return R
ref = load_preds(ROOT / "results/dev" / sys.argv[2].split(":")[0], sys.argv[2].split(":")[1], n)
mem = [load_preds(ROOT / "results/dev" / a.split(":")[0], a.split(":")[1], n) for a in sys.argv[3:]]
Y, C = ref[2], ref[3]
for m in mem:
    assert (m[2] == Y).all() and (m[3] == C).all()
E = np.mean([ranks(m[0], C) for m in mem], axis=0)[None]
P = (E > 0.5).astype(np.int8)          # thresholds irrelevant for AUROC/AP; F1 not reported
r = metrics.report(E, Y, P, C, K, n_boot=500)
Rref = ranks(ref[0], C)[None]          # the reference is rank-normalised the same way
d = metrics.paired_delta((E, P), (Rref, (Rref > 0.5).astype(np.int8)), Y, C, n_boot=500)
names = "+".join(a.split(":")[1] for a in sys.argv[3:])
print(f"n={n} ENSEMBLE[{names}] AUROC {metrics.fmt(r['auroc_mean'])} mAP {r['macro_ap']['point']:.3f} | "
      + " ".join(f"{k}:{r['per_species'][k]['auroc']:.2f}" for k in K)
      + f" | vs {sys.argv[2].split(':')[1]} dAUROC {metrics.fmt(d['auroc_mean'])} P>0 {d['auroc_mean']['p_gt0']:.2f}", flush=True)
