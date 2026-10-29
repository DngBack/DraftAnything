#!/usr/bin/env python3
"""BBBC041 analyses with instance ground truth (unit = image, bootstrap over images).

1. Context leakage at fixed own evidence: per class k, least squares over out-of-fold images
       logit(score_k) ~ a + b_own * y_k + c * log1p(count_k) + sum_{j != k} B[k, j] y_j
   B[k, j] measures how the score for k moves with class j present, holding k's presence and
   its number of annotated cells fixed. 95 % intervals from 2000 image resamples.
2. Tile localisation (methods saved with save_tile_evidence): AUROC of the per-tile evidence for k
   against "a k box centre lies in this tile", over all evaluation tiles, and the same restricted to
   images that contain k (does evidence sit on the k cells, not just on k-positive images?).

    python scripts/analyze_instances.py --methods mean_lin meanstd_lin deepsets pe_quantile pe_quantile_tile1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.metrics import auroc  # noqa: E402

CLASSES = {"bbbc041": ["gametocyte", "leukocyte", "ring", "schizont", "trophozoite"],
           "pannuke": ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]}
DS = "bbbc041"
K = CLASSES[DS]


def fit(L, y, cnt, k):
    others = [j for j in range(y.shape[1]) if j != k]
    X = np.c_[np.ones(len(y)), y[:, k], np.log1p(cnt[:, k]), y[:, others]]
    return np.linalg.lstsq(X, L[:, k], rcond=None)[0]


def recompute_rows() -> np.ndarray:
    """Evaluation bags are deterministic (bag_seed 0): rebuild their store rows as run_dev does."""
    from popev.bags import eval_bags
    from popev.phoebi import Manifest
    man = Manifest(ROOT / f"data/{DS}/splits_lco.json", ROOT / f"data/{DS}/lco_dev_folds.json")
    rows = []
    for f in range(len(man.folds)):
        sp = man.fold(f); pos, _, _ = eval_bags(sp.eval, 1, 0, 1)
        rows.append(sp.eval.rows[pos[:, 0]])
    return np.concatenate(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CLASSES), default="bbbc041")
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    global DS, K
    DS, K = a.dataset, CLASSES[a.dataset]
    a.run = a.run or ROOT / f"results/dev/{DS}"
    from popev.phoebi import Manifest
    man = Manifest(ROOT / f"data/{DS}/splits_lco.json")
    lab = np.zeros((len(man.paths), len(K)), int)
    for v in man.parts.values():
        lab[v.rows] = v.labels
    np.save(ROOT / f"data/{DS}/labels_by_row.npy", lab)
    TL = np.load(ROOT / f"cache/tile_labels_{DS}.npy"); IC = np.load(ROOT / f"cache/instance_counts_{DS}.npy")
    out = [f"# {DS} analyses (unit = image, out-of-fold)", ""]
    for m in a.methods:
        f = a.run / f"preds_{m}_n1.npz"
        if not f.exists():
            out += [f"## {m}: no predictions", ""]; continue
        d = np.load(f); S = d["S"].mean(0).clip(1e-4, 1 - 1e-4); Y = d["Y"].astype(float)
        rows = d["R"][:, 0] if "R" in d.files else recompute_rows()
        assert (Y == np.load(ROOT / f"data/{DS}/labels_by_row.npy")[rows]).all(), "row mapping mismatch"
        cnt = IC[rows].astype(float); L = np.log(S / (1 - S))
        rng = np.random.default_rng(0)
        out += [f"## {m}", "", "Context leakage B[k, j] (score of k when j present, k presence and count held fixed), "
                "with 95 % image-bootstrap intervals; * = interval excludes 0:", "",
                "| score of \\ present | " + " | ".join(K) + " | own (y_k) |", "|---|" + "---|" * (len(K) + 1)]
        n_sig = 0; tot = 0
        for k in range(len(K)):
            b = fit(L, Y, cnt, k); others = [j for j in range(len(K)) if j != k]
            bs = []
            for _ in range(a.n_boot):
                ii = rng.integers(0, len(Y), len(Y))
                if Y[ii, k].min() == Y[ii, k].max():
                    continue
                bs.append(fit(L[ii], Y[ii], cnt[ii], k))
            bs = np.asarray(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            cells = []
            for j in range(len(K)):
                if j == k:
                    cells.append("—"); continue
                c = 3 + others.index(j); sig = lo[c] > 0 or hi[c] < 0; n_sig += sig; tot += 1
                cells.append(f"{b[c]:+.2f} [{lo[c]:+.2f}, {hi[c]:+.2f}]{'*' if sig else ''}")
            out.append(f"| {K[k]} | " + " | ".join(cells) + f" | {b[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}] |")
        out += ["", f"Context coefficients resolved (CI excludes 0): {n_sig}/{tot}", ""]
        if "A" in d.files:
            A = d["A"].astype(np.float32).mean(0)                       # (M, T, K)
            G = TL[rows]                                               # (M, T, K)
            out += ["Tile localisation AUROC (evidence vs box-centre-in-tile):", "",
                    "| class | all tiles | tiles of images containing the class |", "|---|---|---|"]
            brng = np.random.default_rng(1)
            for k in range(len(K)):
                pos_img = Y[:, k] > 0
                bs = []
                for _ in range(min(a.n_boot, 300)):          # image resamples; tiles of an image stay together
                    ii = brng.integers(0, len(Y), len(Y))
                    v = auroc(A[ii][..., k].ravel(), G[ii][..., k].ravel())
                    if v == v:
                        bs.append(v)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                out.append(f"| {K[k]} | {auroc(A[..., k].ravel(), G[..., k].ravel()):.3f} [{lo:.3f}, {hi:.3f}] | "
                           f"{auroc(A[pos_img][..., k].ravel(), G[pos_img][..., k].ravel()):.3f} |")
            out.append("")
    p = ROOT / f"results/analysis/{DS}_{a.run.name}.md"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out)); print("\n".join(out))


if __name__ == "__main__":
    main()
