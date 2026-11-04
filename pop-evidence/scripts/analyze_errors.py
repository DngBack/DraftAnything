#!/usr/bin/env python3
"""Stage B failure analysis on pooled dev predictions (plan §11).

1. Per-combination table: ground truth, per-species mean score and prediction rate.
2. Context attribution: for each species k, least squares over dev combinations
       logit(mean score_k | c) ~ a_k + sum_j B[k, j] y_{c, j}
   B[k, k] is evidence for k itself; B[k, j], j != k, is how much the score for k moves with
   species j being present (context leakage). Reported: B, the own/context ratio, and the
   correlation between B's off-diagonal and the phi co-occurrence of the trained pool.
3. Pure-culture discriminability on the trained pool is left to scripts/ for later (needs train preds).

    python scripts/analyze_errors.py --run results/dev/probe_cls --methods mean_lin meanstd_lin deepsets --n 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.experiment import load_preds  # noqa: E402
from popev.metrics import by_combination  # noqa: E402
from popev.phoebi import PHOEBI_ROOT, Manifest, parse_combo  # noqa: E402


def attribution(S: np.ndarray, Y: np.ndarray, C: np.ndarray):
    uniq = sorted(set(C))
    m = np.stack([S[C == c].mean(0) for c in uniq]).clip(1e-4, 1 - 1e-4)
    y = np.stack([Y[C == c][0] for c in uniq]).astype(float)
    L = np.log(m / (1 - m))
    X = np.c_[np.ones(len(uniq)), y]
    B = np.linalg.lstsq(X, L, rcond=None)[0][1:].T          # (K, K): row k = score of k, column j = presence of j
    return B


def phi(M: np.ndarray) -> np.ndarray:
    return np.corrcoef(M.T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--data_dir", type=Path, default=PHOEBI_ROOT / "data")
    a = ap.parse_args()
    man = Manifest(a.data_dir / "splits_lco.json")
    K = man.class_names
    Ptrain = phi(np.array([[k in parse_combo(c) for k in K] for c in man.trained], dtype=float))
    off = ~np.eye(len(K), dtype=bool)
    L = [f"# Failure analysis — {a.run.name}, n_fields={a.n}", "",
         "Context attribution B[k, j]: change in logit(mean score of k) when species j is present, least squares over "
         "the 26 pooled dev combinations (26 points, 7 parameters per row: descriptive, not inferential). "
         "Diagonal = own evidence, off-diagonal = context. `r(coocc)` = correlation of a row's off-diagonal with the "
         "phi co-occurrence of species k in the 31 trained combinations.", ""]
    for m in a.methods:
        S, P, Y, C, _ = load_preds(a.run, m, a.n)
        Sm = S.mean(0)
        B = attribution(Sm, Y, C)
        L += [f"## {m}", "", "| score of \\ present | " + " | ".join(K) + " | own | ‖context‖₁ | r(coocc) |",
              "|---|" + "---|" * (len(K) + 3)]
        for i, k in enumerate(K):
            ctx = B[i, off[i]]
            r = np.corrcoef(ctx, Ptrain[i, off[i]])[0, 1]
            L.append(f"| {k} | " + " | ".join(f"**{B[i, j]:+.2f}**" if i == j else f"{B[i, j]:+.2f}" for j in range(len(K)))
                     + f" | {B[i, i]:+.2f} | {np.abs(ctx).sum():.2f} | {r:+.2f} |")
        allr = np.corrcoef(B[off], Ptrain[off])[0, 1]
        L += ["", f"Pooled over species: corr(off-diagonal B, train co-occurrence phi) = {allr:+.2f}; "
              f"mean own = {np.diag(B).mean():+.2f}, mean |context| = {np.abs(B[off]).mean():.2f}", ""]
        rows = by_combination(Sm, Y, P[0], C, K)
        L += ["Per combination (mean score; ✓ present):", "", "| combo | " + " | ".join(K) + " | F1 |",
              "|---|" + "---|" * (len(K) + 1)]
        for r in rows:
            g = set(parse_combo(r["combo"]))
            L.append(f"| {r['combo']} | " + " | ".join(f"{'✓' if k in g else ' '}{r['mean_score'][k]:.2f}" for k in K)
                     + f" | {r['f1']:.2f} |")
        L.append("")
    out = ROOT / "results/analysis" / f"errors_{a.run.name}_n{a.n}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
