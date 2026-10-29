"""Bags of fields: n images of one combination, pooled into a set of n*16 tiles.

In PHOEBI a combination is one video, so a bag is n frames-crops of the same acquisition. Every
method in a comparison gets the same n; n=1 is the paper's per-image setting.
Evaluation bags are disjoint and deterministic (seeded shuffle per combination, remainder dropped,
optional cap per combination); training bags are resampled every step, combinations uniformly.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from popev.phoebi import View


def eval_bags(view: View, n: int, seed: int, cap: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (positions into view (M, n), labels (M, K), combos (M,))."""
    rng = np.random.default_rng(seed)
    out, labs, cmb = [], [], []
    for c in sorted(set(view.combos)):
        idx = np.where(view.combos == c)[0]
        idx = idx[rng.permutation(len(idx))]
        m = len(idx) // n
        if cap is not None:
            m = min(m, cap)
        if m == 0:
            continue
        out.append(idx[: m * n].reshape(m, n))
        labs.append(np.repeat(view.labels[idx[:1]], m, axis=0))
        cmb.append(np.full(m, c))
    return np.concatenate(out), np.concatenate(labs), np.concatenate(cmb)


class BagSampler:
    """Training bags: pick a combination uniformly, then n of its images without replacement."""

    def __init__(self, view: View, n: int, seed: int) -> None:
        self.n, self.rng = n, np.random.default_rng(seed)
        self.combos = sorted(set(view.combos))
        self.by_combo: Dict[str, np.ndarray] = {c: np.where(view.combos == c)[0] for c in self.combos}
        self.label_of = {c: view.labels[self.by_combo[c][0]] for c in self.combos}

    def sample(self, b: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cs = self.rng.choice(len(self.combos), size=b)
        pos = np.stack([self.rng.choice(self.by_combo[self.combos[i]], size=self.n, replace=False) for i in cs])
        return pos, np.stack([self.label_of[self.combos[i]] for i in cs]), cs
