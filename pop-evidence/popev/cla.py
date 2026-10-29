"""Contextual logit adjustment (CLA): remove the label-conditional prior from what a set-trained decoder learns.

With set labels, BCE drives sigma(e_k(x)) towards p(y_k = 1 | x). The image x carries the class's own
evidence o_k and the other classes' content c, which reveals y_{-k}. If c is informative about y_k only
through y_{-k}, and o_k is independent of y_{-k} given y_k, then

    logit p(y_k = 1 | x) = LLR_k(o_k) + logit p(y_k = 1 | y_{-k}).

The second term is the context-conditional prior: a function of which other classes are present, learnt
from the training combinations, and wrong whenever combinations shift. CLA trains on e_k + tau * b_k(y_{-k}),
with b_k = logit p_hat(y_k = 1 | y_{-k}) estimated from training labels alone and the other classes' *true*
labels (known at training time), and scores with e_k at test. At the BCE optimum with tau = 1 and a correct
prior, e_k = LLR_k(o_k): own evidence only, invariant to the label-combination distribution.

This is logit adjustment (Menon et al., ICLR 2021) with the marginal prior log pi_y replaced by a
label-conditional one. mode = "marginal" keeps only the intercept (multi-label logit adjustment) as the
ablation that separates the conditional part from plain prior correction.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class ContextPrior:
    def __init__(self, Y: np.ndarray, mode: str = "conditional", C: float = 1.0, eps: float = 1e-3,
                 Z: np.ndarray | None = None) -> None:
        """Y: (N, K) training label vectors, one row per training unit (combination or image).
        Z: optional (N, Q) known covariates of each unit (e.g. one-hot tissue type). With Z the prior is
        p(y_k | y_{-k}, z): the offset then also absorbs what the covariate predicts about y_k, so the
        decoder is not rewarded for encoding a confounder that drives both labels and appearance."""
        Y = np.asarray(Y, dtype=int)
        self.K, self.mode = Y.shape[1], mode
        self.W = np.zeros((self.K, self.K)); self.b = np.zeros(self.K)
        self.V = np.zeros((self.K, 0 if Z is None else Z.shape[1]))
        for k in range(self.K):
            p = np.clip(Y[:, k].mean(), eps, 1 - eps)
            self.b[k] = np.log(p / (1 - p))
            if mode != "conditional" or Y[:, k].min() == Y[:, k].max():
                continue
            others = [j for j in range(self.K) if j != k]
            X = Y[:, others] if Z is None else np.c_[Y[:, others], Z]
            lr = LogisticRegression(C=C, max_iter=2000).fit(X, Y[:, k])
            self.W[k, others] = lr.coef_[0][: len(others)]; self.b[k] = lr.intercept_[0]
            if Z is not None:
                self.V[k] = lr.coef_[0][len(others):]

    def logits(self, y: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """y: (B, K) true labels [, z: (B, Q) covariates] -> (B, K) prior logits (W has a zero diagonal)."""
        W = torch.as_tensor(self.W, dtype=y.dtype, device=y.device)
        b = torch.as_tensor(self.b, dtype=y.dtype, device=y.device)
        out = y @ W.T + b
        if z is not None and self.V.shape[1]:
            out = out + z @ torch.as_tensor(self.V, dtype=y.dtype, device=y.device).T
        return out


def unit_labels(view) -> np.ndarray:
    """One label vector per grouping unit, matching the sampler (combinations drawn uniformly)."""
    _, first = np.unique(view.combos, return_index=True)
    return view.labels[np.sort(first)]
