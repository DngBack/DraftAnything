#!/usr/bin/env python3
"""Episodic Prototype MIL (EP-MIL), feasibility run (docs/research_directions.md, option A).

Learn a projection g so that a tile's evidence for class k is its cosine to the class prototype computed from
*same-batch* reference (pure-culture) images, g applied to both. Image score = tau * mean_t cos(g(z_t), p_k) + beta,
with tau and beta shared scalars (no per-class weights or thresholds, which would carry batch-specific shortcuts).

Training on phoebi6 (random-split train, all 40 combinations), in episodes: each step samples a batch of bags and,
for every class, R reference images from that class's pure culture; an episode-level random feature shift
(translation + mild linear mixing) is applied identically to references and bags, so g learns comparisons that
survive a shift shared by both. Loss = BCE on image scores + exact tile negatives for absent classes.

Evaluation (EXPLORATORY, phoebi4 already read): phoebi4 mixture videos, prototypes from phoebi4 pure-culture
videos (same batch) and, for contrast, from phoebi6 (other batch). 'identity' = g untrained (closed form).

    python scripts/epmil.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from popev.features import open_store  # noqa: E402
from popev.metrics import auroc  # noqa: E402
from popev.phoebi import Manifest  # noqa: E402


class Proj(nn.Module):
    def __init__(self, D=384, H=256, O=128, identity=False):
        super().__init__()
        self.identity = identity
        self.net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, O))
        self.tau = nn.Parameter(torch.tensor(10.0)); self.beta = nn.Parameter(torch.tensor(-5.0))

    def forward(self, z):
        return F.normalize(z if self.identity else self.net(z), dim=-1)


def prototypes(g, refs):
    """refs: list over classes of (R, T, D) tiles -> (K, O) unit prototypes."""
    return torch.stack([F.normalize(g(r).flatten(0, 1).mean(0), dim=0) for r in refs])


def score(g, x, P):
    """x: (B, T, D) -> image logits (B, K), tile logits (B, T, K)."""
    s = g(x) @ P.T
    t = g.tau * s + g.beta
    return t.mean(1), t


def shift(gen, D, dev, sb, sm):
    """Episode-level feature shift shared by references and bags."""
    b = torch.randn(D, generator=gen).to(dev) * sb / D ** 0.5
    M = torch.eye(D, device=dev) + torch.randn(D, D, generator=gen).to(dev) * sm / D ** 0.5
    return lambda z: F.normalize((z + b) @ M, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--R", type=int, default=8); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lam_tile", type=float, default=1.0); ap.add_argument("--sb", type=float, default=0.5)
    ap.add_argument("--sm", type=float, default=0.3); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(); dev = torch.device(a.device)
    paths = json.loads((ROOT / "data/phoebi4/store_paths.json").read_text())
    man = Manifest(ROOT / "data/phoebi4/splits_lco.json", store_paths=paths)
    Fs = open_store(man.paths, "cls", "_p4"); K6 = man.class_names; K4 = ["bs", "fj", "ka", "pf"]; J4 = [K6.index(k) for k in K4]
    tr, te = man.parts["train"], man.parts["test"]

    def load(rows):
        o = np.argsort(rows); x = np.empty((len(rows),) + Fs.shape[1:], np.float16); x[o] = Fs[rows[o]]
        return torch.from_numpy(x).to(dev)
    Xtr = load(tr.rows); Ytr = torch.from_numpy(tr.labels).float().to(dev); Ctr = tr.combos
    pure_idx = [np.where(Ctr == k)[0] for k in K6]                     # every phoebi6 class has a pure culture here
    combos = sorted(set(Ctr)); by_c = {c: np.where(Ctr == c)[0] for c in combos}
    Xte = load(te.rows); Yte = te.labels[:, J4]; Cte = te.combos; mix = Yte.sum(1) >= 2
    ref4 = [np.where(Cte == f"p4_{k[0]}")[0] for k in K4]
    Xm = Xte[torch.from_numpy(np.where(mix)[0]).to(dev)]; Ym = Yte[mix]; Cm = Cte[mix]

    def evaluate(g, refs_x):
        with torch.no_grad():
            P = prototypes(g, refs_x)
            S = torch.cat([score(g, Xm[i:i + 512].float(), P)[0] for i in range(0, len(Xm), 512)]).cpu().numpy()
        return S

    def summary(name, Ss):
        S = np.mean(Ss, axis=0); uniq = sorted(set(Cm)); rng = np.random.default_rng(0); bs = []
        for _ in range(1000):
            ii = np.concatenate([np.where(Cm == u)[0] for u in rng.choice(uniq, len(uniq))])
            bs.append(np.nanmean([auroc(S[ii, j], Ym[ii, j]) for j in range(4)]))
        au = [auroc(S[:, j], Ym[:, j]) for j in range(4)]; lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return f"| {name} | {np.nanmean(au):.3f} [{lo:.3f}, {hi:.3f}] | " + " | ".join(f"{v:.2f}" for v in au) + " |"

    refs_same = [Xte[torch.from_numpy(r).to(dev)].float() for r in ref4]
    refs_other = [Xtr[torch.from_numpy(pure_idx[K6.index(k)][:600]).to(dev)].float() for k in K4]
    rows = ["# EP-MIL feasibility (EXPLORATORY; train phoebi6, test phoebi4 mixtures, 10 videos)", "",
            f"steps {a.steps}, R {a.R}, batch {a.batch}, lambda_tile {a.lam_tile}, shift sb {a.sb} sm {a.sm}, seeds {a.seeds}", "",
            "| model / references | mean AUROC [95% CI] | bs | fj | ka | pf |", "|---|---|---|---|---|---|"]
    gid = Proj(identity=True).to(dev)
    rows.append(summary("identity g (closed form), same-batch refs", [evaluate(gid, refs_same)]))
    rows.append(summary("identity g (closed form), other-batch refs", [evaluate(gid, refs_other)]))
    for variant, (sb, sm) in {"EP-MIL, episode shift": (a.sb, a.sm), "EP-MIL, no shift": (0.0, 0.0)}.items():
        S_same, S_other = [], []
        for seed in a.seeds:
            torch.manual_seed(seed); rng = np.random.default_rng(seed); gen = torch.Generator().manual_seed(seed)
            g = Proj().to(dev); opt = torch.optim.AdamW(g.parameters(), 1e-3, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
            for step in range(a.steps):
                aug = shift(gen, Xtr.shape[-1], dev, sb, sm) if (sb or sm) else (lambda z: z)
                refs = [aug(Xtr[torch.from_numpy(rng.choice(pi, a.R, replace=False)).to(dev)].float()) for pi in pure_idx]
                cs = rng.choice(len(combos), a.batch); idx = np.array([rng.choice(by_c[combos[c]]) for c in cs])
                it = torch.from_numpy(idx).to(dev); x = aug(Xtr[it].float()); y = Ytr[it]
                P = prototypes(g, refs); e, t = score(g, x, P)
                loss = F.binary_cross_entropy_with_logits(e, y) + a.lam_tile * ((1 - y) * F.softplus(t).mean(1)).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            g.eval(); S_same.append(evaluate(g, refs_same)); S_other.append(evaluate(g, refs_other))
            print(f"{variant} seed {seed} done, loss {loss.item():.3f}", flush=True)
        rows.append(summary(f"{variant}, same-batch refs", S_same))
        rows.append(summary(f"{variant}, other-batch refs", S_other))
    out = ROOT / "results/analysis/epmil_feasibility.md"; out.write_text("\n".join(rows)); print("\n".join(rows))


if __name__ == "__main__":
    main()
