#!/usr/bin/env python3
"""EXPLORATORY (phoebi4 read): closed-form, same-batch scoring variants on the phoebi4 mixtures.
No parameter is fitted on another batch. References = phoebi4 pure-culture videos.
  mean_cos   mean over tiles of cos(tile, class prototype)                      [0.764 reference]
  margin     mean over tiles of cos_k - max_{j!=k} cos_j
  knn10      mean over tiles of the mean cosine to the 10 nearest reference tiles of class k
  refine1    prototypes updated once with mixture tiles softly assigned (temperature 0.05) to the classes,
             transductive within the batch (no labels)
Unit = video; 95% CI by video bootstrap."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
import torch.nn.functional as F
from popev.features import open_store
from popev.phoebi import Manifest
from popev.metrics import auroc
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = json.loads((ROOT / "data/phoebi4/store_paths.json").read_text())
man = Manifest(ROOT / "data/phoebi4/splits_lco.json", store_paths=paths)
Fs = open_store(man.paths, "cls", "_p4"); K6 = man.class_names; K4 = ["bs", "fj", "ka", "pf"]; J4 = [K6.index(k) for k in K4]
te = man.parts["test"]
o = np.argsort(te.rows); X = np.empty((len(te.rows),) + Fs.shape[1:], np.float16); X[o] = Fs[te.rows[o]]
X = F.normalize(torch.from_numpy(X).float().to(dev), dim=-1)
Y = te.labels[:, J4]; C = te.combos; mix = Y.sum(1) >= 2
refs = [X[torch.from_numpy(np.where(C == f"p4_{k[0]}")[0]).to(dev)].flatten(0, 1) for k in K4]
Xm = X[torch.from_numpy(np.where(mix)[0]).to(dev)]; Ym, Cm = Y[mix], C[mix]
P = F.normalize(torch.stack([r.mean(0) for r in refs]), dim=1)
def mean_cos(P): return (Xm @ P.T).mean(1)
def margin(P):
    s = Xm @ P.T; out = []
    for k in range(4):
        other = torch.cat([s[..., :k], s[..., k + 1:]], -1).amax(-1); out.append((s[..., k] - other).mean(1))
    return torch.stack(out, 1)
def knn(k=10):
    out = []
    for r in refs:
        sub = r[torch.randperm(len(r), generator=torch.Generator().manual_seed(0))[:8000].to(dev)]
        sims = []
        for i in range(0, len(Xm), 256):
            s = Xm[i:i + 256].flatten(0, 1) @ sub.T; sims.append(s.topk(k, dim=1).values.mean(1).view(-1, Xm.shape[1]).mean(1))
        out.append(torch.cat(sims))
    return torch.stack(out, 1)
def refine(P, T=0.05):
    t = Xm.flatten(0, 1); w = torch.softmax((t @ P.T) / T, dim=1)                 # soft assignment of mixture tiles
    Pm = F.normalize((w.T @ t) / w.sum(0)[:, None], dim=1)
    return F.normalize(P + Pm, dim=1)
scores = {"mean_cos": mean_cos(P), "margin": margin(P), "knn10": knn(10), "refine1": mean_cos(refine(P))}
uniq = sorted(set(Cm)); rng = np.random.default_rng(0)
rows = ["# Same-batch closed-form variants (EXPLORATORY; phoebi4 mixtures)", "", "| scorer | mean AUROC [95% CI] | bs | fj | ka | pf |", "|---|---|---|---|---|---|"]
for name, S in scores.items():
    S = S.cpu().numpy(); au = [auroc(S[:, j], Ym[:, j]) for j in range(4)]; bs = []
    for _ in range(1000):
        ii = np.concatenate([np.where(Cm == u)[0] for u in rng.choice(uniq, len(uniq))]); bs.append(np.nanmean([auroc(S[ii, j], Ym[ii, j]) for j in range(4)]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); rows.append(f"| {name} | {np.nanmean(au):.3f} [{lo:.3f}, {hi:.3f}] | " + " | ".join(f"{v:.2f}" for v in au) + " |")
(ROOT / "results/analysis/inbatch_variants.md").write_text("\n".join(rows)); print("\n".join(rows))
