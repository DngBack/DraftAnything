#!/usr/bin/env python3
"""Compare methods on an instance dataset: in-distribution vs decorrelated AUROC (paired), context share,
localisation. Usage: compare_inst.py DATASET run:method[:label] ... (first = reference)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import importlib.util
import numpy as np
from popev import metrics
spec = importlib.util.spec_from_file_location("ai", ROOT / "scripts/analyze_instances.py"); ai = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai)
ds = sys.argv[1]; ai.DS, ai.K = ds, ai.CLASSES[ds]
TL = np.load(ROOT / f"cache/tile_labels_{ds}.npy"); IC = np.load(ROOT / f"cache/instance_counts_{ds}.npy")
rows_fb = None
def load(run, m):
    global rows_fb
    z = np.load(ROOT / f"results/dev/{run}/preds_{m}_n1.npz")
    if "R" in z.files: r = z["R"][:, 0]
    else:
        rows_fb = rows_fb if rows_fb is not None else ai.recompute_rows(); r = rows_fb
    return z, r
def share(S, Y, rr):
    Sm = S.mean(0).clip(1e-4, 1 - 1e-4); L = np.log(Sm / (1 - Sm)); cnt = IC[rr].astype(float)
    return np.mean([np.abs(ai.fit(L, Y.astype(float), cnt, k)[3:]).mean() for k in range(Y.shape[1])]) / L.std(0).mean()
def dec_auroc(S, Y, P, w):
    return metrics._Prepared(metrics._as3(S), Y, metrics._as3(P)).seeds(w)["auroc_mean"]
specs = [a.split(":") for a in sys.argv[2:]]
ref = None; rng = np.random.default_rng(0)
for sp in specs:
    run, m = sp[0], sp[1]; z, rr = load(run, m); S, P, Y = z["S"], z["P"], z["Y"]
    w = metrics.decorrelation_weights(Y); one = np.ones(len(Y))
    a_in, a_dec = dec_auroc(S, Y, P, one), dec_auroc(S, Y, P, w)
    line = f"{m:28s} AUROC {a_in:.3f}  decorrelated {a_dec:.3f}  ctx-share {share(S, Y, rr):.2f}"
    if ref is not None:
        pa, pb = metrics._Prepared(metrics._as3(S), Y, metrics._as3(P)), metrics._Prepared(metrics._as3(ref[0]), Y, metrics._as3(ref[1]))
        U = np.arange(len(Y)); d_in, d_dec = [], []
        for _ in range(300):
            ii = np.bincount(rng.integers(0, len(Y), len(Y)), minlength=len(Y)).astype(float)
            d_in.append(pa.seeds(ii)["auroc_mean"] - pb.seeds(ii)["auroc_mean"]); d_dec.append(pa.seeds(ii * w)["auroc_mean"] - pb.seeds(ii * w)["auroc_mean"])
        f = lambda v: f"{np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
        line += f" | Δin {f(d_in)} Δdec {f(d_dec)}"
    if "A" in z.files:
        A = z["A"].astype(np.float32).mean(0); G = TL[rr]
        line += " | loc " + " ".join(f"{metrics.auroc(A[..., k].ravel(), G[..., k].ravel()):.2f}" for k in range(Y.shape[1]))
    print(line, flush=True)
    if ref is None: ref = (S, P)
