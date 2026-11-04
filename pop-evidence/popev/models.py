"""Bag decoders over frozen tile embeddings. Input x: (B, T, D); output logits (B, K).

LinearProbe   linear head on a fixed representation (popev.pooling), standardised on train.
DeepSets      rho(mean_i phi(z_i)) -> K; one shared bag vector for all species.
PopEvidence   shared projection -> per-species evidence a_{i,k} in R^m for every tile ->
              per-species distribution pooling over tiles -> per-species score e_k.
              With class_specific=True, e_k reads only species k's pooled evidence; with False
              the pooled evidence of all species feeds every score (the shared-head ablation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QS = (0.1, 0.25, 0.5, 0.75, 0.9)


class LinearProbe(nn.Module):
    def __init__(self, mu: torch.Tensor, sd: torch.Tensor, K: int) -> None:
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.head = nn.Linear(len(mu), K)

    def forward(self, h):
        return self.head((h - self.mu) / self.sd)


def mlp(d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out))


class DeepSets(nn.Module):
    def __init__(self, D: int, K: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.phi = mlp(D, hidden, hidden, dropout)
        self.rho = mlp(hidden, hidden, K, dropout)

    def forward(self, x):
        return self.rho(F.gelu(self.phi(x)).mean(1)), None


def pool_stats(a: torch.Tensor, pool: str) -> torch.Tensor:
    """a: (B, T, K, m) -> (B, K, m*S) statistics over tiles."""
    mean = a.mean(1)
    if pool == "mean":
        return mean
    std = (a.var(1, unbiased=False) + 1e-6).sqrt()
    if pool == "meanstd":
        return torch.cat([mean, std], -1)
    if pool == "quantile":
        q = torch.quantile(a, torch.tensor(QS, device=a.device, dtype=a.dtype), dim=1)   # (S, B, K, m)
        return torch.cat([mean, std, q.permute(1, 2, 0, 3).flatten(2)], -1)
    if pool == "max":
        return torch.cat([mean, a.amax(1)], -1)
    raise ValueError(pool)


def n_stats(pool: str) -> int:
    return {"mean": 1, "meanstd": 2, "quantile": 2 + len(QS), "max": 2}[pool]


class PopEvidence(nn.Module):
    def __init__(self, D: int, K: int, hidden: int = 256, m: int = 4, pool: str = "meanstd",
                 class_specific: bool = True, dropout: float = 0.1, score: str = "head") -> None:
        """score: "head" = learned read-out of pooled statistics; "max" / "q90" = the max / 0.9-quantile of
        the first evidence channel over tiles, i.e. a score that is a function of tile evidence only."""
        super().__init__()
        self.K, self.m, self.pool, self.class_specific, self.score = K, m, pool, class_specific, score
        self.proj = mlp(D, hidden, hidden, dropout)
        self.evidence = nn.Linear(hidden, K * m)
        S = n_stats(pool) * m
        if class_specific:
            self.w = nn.Parameter(torch.zeros(K, S))
            nn.init.normal_(self.w, std=S ** -0.5)
            self.b = nn.Parameter(torch.zeros(K))
        else:
            self.head = nn.Linear(K * S, K)

    def forward(self, x):
        B, T, _ = x.shape
        a = self.evidence(F.gelu(self.proj(x))).view(B, T, self.K, self.m)
        if self.score == "max":
            return a[..., 0].amax(1), a
        if self.score == "mean":
            return a[..., 0].mean(1), a
        if self.score.startswith("q"):                     # "q90", "q50", ...: that quantile of tile evidence
            return torch.quantile(a[..., 0], int(self.score[1:]) / 100, dim=1), a
        s = pool_stats(a, self.pool)                                     # (B, K, S)
        if self.class_specific:
            e = (s * self.w).sum(-1) + self.b
        else:
            e = self.head(s.flatten(1))
        return e, a


def tile_evidence_loss(a: torch.Tensor, y: torch.Tensor, q: float = 0.9, use_pure: bool = True) -> torch.Tensor:
    """Label-valid tile-level supervision of the first evidence channel a[..., 0] (B, T, K):
    absent species (y_k = 0): no tile may carry evidence for k (every tile pushed down);
    present species: the q-quantile tile must carry it (standard multi-instance assumption);
    pure cultures (one species present): every tile carries it.
    None of these targets depends on which other species are present."""
    t = a[..., 0]
    neg = F.softplus(t).mean(1)                                          # (B, K)
    top = F.softplus(-torch.quantile(t, q, dim=1))                       # (B, K)
    # use_pure = False drops the pure-culture term: single-species bags are then treated like any other
    # positive bag (top-quantile constraint), so no species' evidence depends on having a pure culture
    pure = (y.sum(1, keepdim=True) == 1).float() * float(use_pure)
    allpos = F.softplus(-t).mean(1)
    pos = pure * allpos + (1 - pure) * top
    return (y * pos + (1 - y) * neg).mean()


def context_consistency(e: torch.Tensor, y: torch.Tensor, combo: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """For each species k: variance, across the distinct positive combinations in the batch, of the
    per-combination mean logit e_k (context term), plus a pairwise softplus ranking term that keeps
    positives above negatives (stops the context term from being satisfied by a constant score).
    Applied only where >= 2 positive combinations and >= 1 negative are present."""
    terms = []
    for k in range(e.shape[1]):
        pos, neg = y[:, k] > 0.5, y[:, k] < 0.5
        if not neg.any():
            continue
        cpos = combo[pos]
        uniq = cpos.unique()
        if len(uniq) < 2:
            continue
        cm = torch.stack([e[pos, k][cpos == u].mean() for u in uniq])
        rank = F.softplus(margin - (e[pos, k][:, None] - e[neg, k][None, :])).mean()
        terms.append(cm.var(unbiased=False) + rank)
    return torch.stack(terms).mean() if terms else e.sum() * 0.0


class ClassAttentionMIL(nn.Module):
    """Gated attention MIL (Ilse et al. 2018) with one attention head per class, for multi-label bags.
    Tile evidence returned = the per-class attention logits (B, T, K, 1), the quantity usually shown
    as an attention heat map; the tile constraints can be applied to it like to PopEvidence's."""

    def __init__(self, D: int, K: int, hidden: int = 256, att: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.h = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Dropout(dropout))
        self.V, self.U = nn.Linear(hidden, att), nn.Linear(hidden, att)
        self.w = nn.Linear(att, K)
        self.cls = nn.Parameter(torch.zeros(K, hidden)); nn.init.normal_(self.cls, std=hidden ** -0.5)
        self.b = nn.Parameter(torch.zeros(K))

    def forward(self, x):
        h = self.h(x)                                                     # (B, T, H)
        logit = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (B, T, K)
        att = logit.softmax(1)
        bag = torch.einsum("btk,bth->bkh", att, h)                        # (B, K, H)
        e = (bag * self.cls).sum(-1) + self.b
        return e, logit.unsqueeze(-1)
