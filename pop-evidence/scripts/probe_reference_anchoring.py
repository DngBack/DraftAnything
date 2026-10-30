#!/usr/bin/env python3
"""EXPLORATORY (phoebi4 already read): does anchoring on same-batch reference cultures fix the batch shift?
Evaluate on the phoebi4 *mixture* videos only (orders >= 2); the phoebi4 pure-culture videos act as in-batch
references (separate videos/sessions from the mixtures). Compared with references from the other batch (phoebi6).
Scorers (all per image, mean over the 16 tiles):
  proto_cross   cosine to phoebi6 pure-culture prototypes
  proto_inbatch cosine to phoebi4 pure-culture prototypes
  lin_inbatch   4-way tile classifier (logistic) trained on phoebi4 pure-culture tiles, mean tile posterior
  lin_cross     same classifier trained on phoebi6 pure-culture tiles of bs/fj/ka/pf
  tvn_cross     proto_cross after whitening each batch with its own pure-culture tile statistics (TVN-style)
Unit = video; AUROC per species over images, 95% CI by video bootstrap."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
from popev.features import open_store
from popev.phoebi import Manifest
from popev.metrics import auroc
paths = json.loads((ROOT / "data/phoebi4/store_paths.json").read_text())
man = Manifest(ROOT / "data/phoebi4/splits_lco.json", store_paths=paths)
F = open_store(man.paths, "cls", "_p4"); K6 = man.class_names; KS = ["bs", "fj", "ka", "pf"]; J = [K6.index(k) for k in KS]
te = man.parts["test"]; combos = te.combos; Y = te.labels[:, J]
order = (Y.sum(1))
pure_vid = {k: f"p4_{k[0]}" for k in KS}            # b, f, k, p videos
mix = order >= 2
base = Manifest.default(); c6 = np.empty(len(base.paths), dtype=object)
for v in base.parts.values(): c6[v.rows] = v.combos
def tiles(rows, cap=600):
    rows = np.sort(rows)[:: max(1, len(rows) // cap)]
    return torch.from_numpy(np.asarray(F[rows], dtype=np.float32))
ref4 = {k: tiles(te.rows[combos == pure_vid[k]]) for k in KS}
ref6 = {k: tiles(np.where(c6 == k)[0]) for k in KS}
Xm = torch.from_numpy(np.asarray(F[np.sort(te.rows[mix])], dtype=np.float32)); om = np.argsort(np.argsort(te.rows[mix]))
Xm = Xm[torch.from_numpy(om)]; Ym = Y[mix]; Cm = combos[mix]
def proto(ref, X, whiten=None):
    P = torch.stack([torch.nn.functional.normalize(ref[k].flatten(0, 1).mean(0), dim=0) for k in KS])
    if whiten: X = whiten(X); P = torch.nn.functional.normalize(torch.stack([whiten(ref[k]).flatten(0, 1).mean(0) for k in KS]), dim=1)
    return (torch.nn.functional.normalize(X, dim=-1) @ P.T).mean(1).numpy()
def lin(ref, X):
    x = torch.cat([ref[k].flatten(0, 1) for k in KS]); y = torch.cat([torch.full((ref[k].shape[0] * ref[k].shape[1],), i) for i, k in enumerate(KS)])
    mu, sd = x.mean(0), x.std(0) + 1e-6; W = torch.nn.Linear(x.shape[1], 4); opt = torch.optim.AdamW(W.parameters(), 1e-2, weight_decay=1e-2)
    for _ in range(300):
        l = torch.nn.functional.cross_entropy(W((x - mu) / sd), y); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad(): return W((X - mu) / sd).softmax(-1).mean(1).numpy()
def tvn(ref):
    x = torch.cat([ref[k].flatten(0, 1) for k in KS]); mu = x.mean(0); U, S, _ = torch.linalg.svd((x - mu).T @ (x - mu) / len(x))
    Wh = U @ torch.diag(1 / torch.sqrt(S + 1e-3)) @ U.T
    return lambda X: (X - mu) @ Wh
w6, w4 = tvn(ref6), tvn(ref4)
scores = {"proto_cross": proto(ref6, Xm), "proto_inbatch": proto(ref4, Xm), "lin_cross": lin(ref6, Xm), "lin_inbatch": lin(ref4, Xm)}
# TVN: each batch whitened with its own reference statistics, prototypes from the other batch (cross) after whitening
P6 = torch.nn.functional.normalize(torch.stack([w6(ref6[k]).flatten(0, 1).mean(0) for k in KS]), dim=1)
scores["tvn_cross"] = (torch.nn.functional.normalize(w4(Xm), dim=-1) @ P6.T).mean(1).numpy()
uniq = sorted(set(Cm)); rng = np.random.default_rng(0)
out = ["# Reference anchoring probe (EXPLORATORY; phoebi4 mixtures, 10 videos)", "", "| scorer | mean AUROC [95% CI over videos] | " + " | ".join(KS) + " |", "|---|---|" + "---|" * 4]
for name, S in scores.items():
    au = [auroc(S[:, j], Ym[:, j]) for j in range(4)]
    bs = []
    for _ in range(1000):
        pick = rng.choice(uniq, len(uniq)); ii = np.concatenate([np.where(Cm == u)[0] for u in pick])
        v = [auroc(S[ii, j], Ym[ii, j]) for j in range(4)]; bs.append(np.nanmean(v))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    out.append(f"| {name} | {np.nanmean(au):.3f} [{lo:.3f}, {hi:.3f}] | " + " | ".join(f"{a:.2f}" for a in au) + " |")
out += ["", f"Mixture videos: {uniq}"]
(ROOT / "results/analysis/reference_anchoring.md").write_text("\n".join(out)); print("\n".join(out))
