#!/usr/bin/env python3
"""EXPLORATORY: reference-relative representation. Each tile is described by its cosine similarity to the
pure-culture prototypes of *its own batch* (4 shared species); an image by the mean and 10/50/90 % quantiles of
those similarities per species (16 features). A multi-label logistic head is trained on phoebi6 images (all
combinations, references = phoebi6 pure cultures) and applied to phoebi4 mixtures (references = phoebi4 pure
cultures). Compared with the same head on raw mean embeddings (= mean_lin restricted to 4 species)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
import torch.nn.functional as Fn
from popev.features import open_store
from popev.phoebi import Manifest
from popev.metrics import auroc
paths = json.loads((ROOT / "data/phoebi4/store_paths.json").read_text())
man = Manifest(ROOT / "data/phoebi4/splits_lco.json", store_paths=paths)
F = open_store(man.paths, "cls", "_p4"); K6 = man.class_names; KS = ["bs", "fj", "ka", "pf"]; J = [K6.index(k) for k in KS]
def load(rows):
    o = np.argsort(rows); x = np.empty((len(rows),) + F.shape[1:], np.float32); x[o] = F[rows[o]]; return torch.from_numpy(x)
tr = man.parts["train"]; te = man.parts["test"]
rng = np.random.default_rng(0); sel = np.sort(rng.choice(len(tr), 30000, replace=False))
Xtr = load(tr.rows[sel]); Ytr = tr.labels[sel][:, J]; Ctr = tr.combos[sel]
Xte = load(te.rows); Yte = te.labels[:, J]; Cte = te.combos; mix = Yte.sum(1) >= 2
def protos(X, C, names):
    return torch.stack([Fn.normalize(X[torch.from_numpy(C == n)].flatten(0, 1).mean(0), dim=0) for n in names])
P6 = protos(Xtr, Ctr, KS); P4 = protos(Xte, Cte, [f"p4_{k[0]}" for k in KS])
def relfeat(X, P):
    s = Fn.normalize(X, dim=-1) @ P.T                                         # (N, T, 4)
    q = torch.quantile(s, torch.tensor([0.1, 0.5, 0.9]), dim=1).permute(1, 2, 0)  # (N, 4, 3)
    return torch.cat([s.mean(1, keepdim=True).transpose(1, 2), q], -1).flatten(1)   # (N, 16)
def fit_eval(Htr, Hte, name):
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6; W = torch.nn.Linear(Htr.shape[1], 4)
    opt = torch.optim.AdamW(W.parameters(), 1e-2, weight_decay=1e-2); y = torch.from_numpy(Ytr).float()
    for _ in range(500):
        l = Fn.binary_cross_entropy_with_logits(W((Htr - mu) / sd), y); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad(): S = torch.sigmoid(W((Hte - mu) / sd)).numpy()
    Sm, Ym, Cm = S[mix], Yte[mix], Cte[mix]; uniq = sorted(set(Cm)); r = np.random.default_rng(0); bs = []
    for _ in range(1000):
        ii = np.concatenate([np.where(Cm == u)[0] for u in r.choice(uniq, len(uniq))])
        bs.append(np.nanmean([auroc(Sm[ii, j], Ym[ii, j]) for j in range(4)]))
    au = [auroc(Sm[:, j], Ym[:, j]) for j in range(4)]; lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return f"| {name} | {np.nanmean(au):.3f} [{lo:.3f}, {hi:.3f}] | " + " | ".join(f"{a:.2f}" for a in au) + " |"
rows = ["# Reference-relative representation probe (EXPLORATORY; train phoebi6, test phoebi4 mixtures)", "",
        "| head on | mean AUROC [95% CI over 10 videos] | bs | fj | ka | pf |", "|---|---|---|---|---|---|"]
rows.append(fit_eval(Xtr.mean(1), Xte.mean(1), "raw mean embedding (mean_lin, 4 species)"))
rows.append(fit_eval(relfeat(Xtr, P6), relfeat(Xte, P6), "similarity to OTHER-batch references (phoebi6 protos at test)"))
rows.append(fit_eval(relfeat(Xtr, P6), relfeat(Xte, P4), "similarity to SAME-batch references (reference-relative)"))
# reference-standardised: z-score the relative features with statistics of that batch's own reference images
ref6 = np.isin(Ctr, KS); ref4 = np.isin(Cte, [f"p4_{k[0]}" for k in KS])
def zs(H, Href): return (H - Href.mean(0)) / (Href.std(0) + 1e-6)
H6, H4 = relfeat(Xtr, P6), relfeat(Xte, P4)
rows.append(fit_eval(zs(H6, H6[torch.from_numpy(ref6)]), zs(H4, H4[torch.from_numpy(ref4)]), "same-batch references + reference-standardised features"))
# scale-free closed form for comparison: fraction of tiles whose nearest same-batch prototype is k
def frac(X, P):
    a = (Fn.normalize(X, dim=-1) @ P.T).argmax(-1); return torch.stack([(a == j).float().mean(1) for j in range(4)], 1)
Sf = frac(Xte, P4).numpy()[mix]; Ym, Cm = Yte[mix], Cte[mix]
rows.append("| (closed form) nearest same-batch prototype, fraction of tiles | " + f"{np.nanmean([auroc(Sf[:, j], Ym[:, j]) for j in range(4)]):.3f} | " + " | ".join(f"{auroc(Sf[:, j], Ym[:, j]):.2f}" for j in range(4)) + " |")
(ROOT / "results/analysis/reference_relative.md").write_text("\n".join(rows)); print("\n".join(rows))
