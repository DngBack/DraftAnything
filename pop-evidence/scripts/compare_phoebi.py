#!/usr/bin/env python3
"""Paired comparisons on the pooled PHOEBI dev folds. Usage: compare_phoebi.py N run:method ... (first = reference)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from popev import metrics
from popev.experiment import load_preds
K = ["bs", "bt", "fj", "ka", "mx", "pf"]; n = int(sys.argv[1]); ref = None
for spec in sys.argv[2:]:
    run, m = spec.split(":"); A = load_preds(ROOT / "results/dev" / run, m, n)
    r = metrics.report(A[0], A[2], A[1], A[3], K, n_boot=300)
    line = (f"n={n} {m:22s} AUROC {metrics.fmt(r['auroc_mean'])} mAP {r['macro_ap']['point']:.3f} F1m {r['f1_margin']['point']:+.3f} | "
            + " ".join(f"{k}:{r['per_species'][k]['auroc']:.2f}" for k in K))
    if ref is not None:
        d = metrics.paired_delta(A[:2], ref[:2], A[2], A[3], n_boot=500)
        line += f" | dAUROC vs ref {metrics.fmt(d['auroc_mean'])} P>0 {d['auroc_mean']['p_gt0']:.2f}"
    else:
        ref = A
    print(line, flush=True)
