"""Fixed (unlearned) set representations of a bag of tile embeddings, for the distribution probe.

Every representation is fitted on training tiles only (PCA basis, kernel bandwidth, random
features) and is permutation-invariant over tiles. Input x: (M, T, D) float tensor.

    mean        mu                                         D
    meanstd     [mu; sigma]                                2D
    meanquant   [mu; q10, q50, q90 of PCA-d projections]   D + 3d
    kme         mean of random Fourier features (RBF)      R
"""
from __future__ import annotations

import torch


class Representation:
    name = "base"

    def fit(self, tiles: torch.Tensor, seed: int) -> "Representation":
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Mean(Representation):
    name = "mean"

    def __call__(self, x):
        return x.mean(1)


class MeanStd(Representation):
    name = "meanstd"

    def __call__(self, x):
        return torch.cat([x.mean(1), (x.var(1, unbiased=False) + 1e-6).sqrt()], 1)


class MeanQuantile(Representation):
    name = "meanquant"

    def __init__(self, d: int = 32, qs=(0.1, 0.5, 0.9)) -> None:
        self.d, self.qs = d, torch.tensor(qs)

    def fit(self, tiles, seed):
        self.mu = tiles.mean(0)
        _, _, v = torch.pca_lowrank(tiles - self.mu, q=self.d, center=False, niter=4)
        self.W = v[:, : self.d]
        return self

    def __call__(self, x):
        p = (x - self.mu.to(x)) @ self.W.to(x)                       # (M, T, d)
        q = torch.quantile(p, self.qs.to(x), dim=1)                    # (3, M, d)
        return torch.cat([x.mean(1), q.permute(1, 0, 2).reshape(len(x), -1)], 1)


class KernelMean(Representation):
    """Empirical kernel mean embedding of the tile distribution, RBF via random Fourier features.
    Bandwidth = scale x median pairwise distance of a training-tile subsample."""
    name = "kme"

    def __init__(self, n_features: int = 2048, scale: float = 1.0) -> None:
        self.R, self.scale = n_features, scale

    def fit(self, tiles, seed):
        g = torch.Generator().manual_seed(seed)
        sub = tiles[torch.randperm(len(tiles), generator=g)[:4000]]
        self.bandwidth = float(torch.cdist(sub, sub).median()) * self.scale
        D = tiles.shape[1]
        self.Om = torch.randn(D, self.R, generator=g) / self.bandwidth
        self.b = torch.rand(self.R, generator=g) * 2 * torch.pi
        return self

    def __call__(self, x):
        z = torch.cos(x @ self.Om.to(x) + self.b.to(x)) * (2.0 / self.R) ** 0.5
        return z.mean(1)


def make(name: str, **kw) -> Representation:
    return {"mean": Mean, "meanstd": MeanStd, "meanquant": MeanQuantile, "kme": KernelMean}[name](**kw)
