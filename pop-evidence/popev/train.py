"""Train one method on one Split and score its calibration and evaluation bags.

Method configs (configs/*.yaml):
  {type: fixed, rep: mean|meanstd|meanquant|kme, rep_kw: {...}, wd, steps, lr}
  {type: set, model: deepsets|popevidence, model_kw: {...}, lambda_cons, steps, lr, wd, batch}
Thresholds are per-species argmax-F1 on the calibration bags (train combinations, val images),
so nothing about the evaluated combinations is seen before scoring.
"""
from __future__ import annotations

from typing import Dict

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from popev import pooling
from popev.cla import ContextPrior, unit_labels
from popev.bags import BagSampler, eval_bags
from popev.metrics import argmax_f1_thresholds
from popev.models import ClassAttentionMIL, DeepSets, LinearProbe, PopEvidence, context_consistency, tile_evidence_loss
from popev.phoebi import Split, View


def load_tiles(store: np.ndarray, view: View) -> torch.Tensor:
    order = np.argsort(view.rows)
    x = np.empty((len(view), *store.shape[1:]), dtype=np.float16)
    x[order] = store[view.rows[order]]
    return torch.from_numpy(x)


def gather(X: torch.Tensor, pos: np.ndarray, device) -> torch.Tensor:
    """(M, n) image positions -> (M, n*T, D) float tiles on device. X may live on the CPU or on the device."""
    x = X[torch.from_numpy(pos.reshape(-1)).to(X.device)].to(device, non_blocking=True).float()
    return x.view(pos.shape[0], -1, X.shape[-1])


def mix_tiles(xa: torch.Tensor, xb: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """CompMix: each tile position of the bag comes from bag A or bag B with probability 1/2 (fixed per
    position, random per call). The mixed bag's label set is y_A | y_B: tiles only contain species of
    their own combination, so the union is label-valid and the mixed bag is a new combination."""
    take_a = torch.rand(xa.shape[1], generator=gen) < 0.5
    take_a = take_a.to(xa.device).view(1, -1, 1)
    return torch.where(take_a, xa, xb)


@torch.no_grad()
def apply_chunked(fn, X, pos, device, chunk: int = 256) -> torch.Tensor:
    return torch.cat([fn(gather(X, pos[i:i + chunk], device)).float().cpu() for i in range(0, len(pos), chunk)])


def run(cfg: Dict, split: Split, store: np.ndarray, n_fields: int, seed: int, device: str = "cuda",
        bag_seed: int = 0, cap_train: int = 600, cap_calib: int = 150, cap_eval: int = 300, log=print) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    if cfg.get("pure_only"):
        # ablation: train only on single-species bags (closest to pure-culture prototypes)
        single = split.train.labels.sum(1) == 1
        split = Split(split.train.select(single), split.calib, split.eval, split.name)
    Xtr, Xca, Xev = (load_tiles(store, v) for v in (split.train, split.calib, split.eval))
    K = split.train.labels.shape[1]
    # evaluation bags depend on bag_seed only, so every method and training seed scores the same bags
    ca_pos, ca_y, ca_c = eval_bags(split.calib, n_fields, bag_seed, cap_calib)
    ev_pos, ev_y, ev_c = eval_bags(split.eval, n_fields, bag_seed, cap_eval)

    cla = cfg.get("cla")
    cov = None
    if cla and cla.get("covariate"):
        # per-row covariate in feature-store order, one-hot encoded (e.g. data/pannuke/tissue.npy)
        raw = np.load(Path(__file__).resolve().parents[1] / cla["covariate"], allow_pickle=True)
        cats = np.unique(raw); cov = (raw[:, None] == cats[None, :]).astype(np.float32)
    if cla:
        _, first = np.unique(split.train.combos, return_index=True); first = np.sort(first)
        Ztr = cov[split.train.rows[first]] if cov is not None else None
        prior = ContextPrior(split.train.labels[first], cla.get("mode", "conditional"), cla.get("C", 1.0), Z=Ztr)
    else:
        prior = None
    tau = cla.get("tau", 1.0) if cla else 0.0

    def adj(y, rows=None):
        if prior is None:
            return 0.0
        z = torch.as_tensor(cov[rows], device=y.device) if (cov is not None and rows is not None) else None
        return tau * prior.logits(y, z)

    if cfg.get("mix") and cov is not None:
        raise ValueError("CompMix with a CLA covariate is not supported: mixed bags have no single covariate value")
    if dev.type == "cuda" and cfg["type"] == "set":
        # keep the training tiles on the GPU (fp16, ~1 GB for PHOEBI): removes host-side indexing per step
        Xtr = Xtr.to(dev)

    if cfg["type"] == "proto":
        # ProtoMatch re-implementation (PHOEBI): prototype of species k = mean tile embedding of its pure
        # culture in the training pool; a species without one (bt) gets the hybrid prototype = mean tile of
        # the training images that contain it. Tile score = cosine to each prototype, image = mean over tiles.
        Xf = Xtr.float(); lab = split.train.labels
        protos = []
        for k in range(K):
            pure = (lab[:, k] == 1) & (lab.sum(1) == 1)
            sel = pure if pure.any() else lab[:, k] == 1
            protos.append(F.normalize(Xf[torch.from_numpy(sel)].flatten(0, 1).mean(0), dim=0))
        Pm = torch.stack(protos).to(dev)
        agg = cfg.get("agg", "mean")
        if agg == "knn":
            # reference instances per class (subsample), for nearest-reference scoring
            gen_k = torch.Generator().manual_seed(0); refs_k = []
            for k in range(K):
                pure = (lab[:, k] == 1) & (lab.sum(1) == 1); sel = pure if pure.any() else lab[:, k] == 1
                r = F.normalize(Xf[torch.from_numpy(sel)].flatten(0, 1), dim=-1)
                refs_k.append(r[torch.randperm(len(r), generator=gen_k)[:8000]].to(dev))
        def score(x):
            xn = F.normalize(x, dim=-1)
            if agg == "knn":
                # mean over instances of the mean cosine to the 10 nearest reference instances of each class
                B, T, D = xn.shape; flat = xn.reshape(-1, D)
                return torch.stack([(flat @ r.T).topk(10, dim=1).values.mean(1).view(B, T).mean(1) for r in refs_k], 1)
            c = xn @ Pm.T                                                    # (B, T, K)
            if agg == "mean":
                return c.mean(1)
            if agg == "vote":
                # fraction of instances whose nearest prototype is k: a cell count per class when instances are cells
                return F.one_hot(c.argmax(-1), K).float().mean(1)
            return torch.quantile(c, float(agg[1:]) / 100, dim=1)
        model = None; n_params = int(Pm.numel())
    elif cfg["type"] == "fixed":
        g = torch.Generator().manual_seed(seed)
        sub = Xtr.flatten(0, 1)[torch.randperm(Xtr.shape[0] * Xtr.shape[1], generator=g)[:50000]].float()
        rep = pooling.make(cfg["rep"], **cfg.get("rep_kw", {})).fit(sub, seed)
        tr_pos, tr_y, _ = eval_bags(split.train, n_fields, seed + 1, cap_train)
        H = apply_chunked(rep, Xtr, tr_pos, dev)
        mix = cfg.get("mix")
        if mix:
            # fixed representations: add len(tr_pos) * ratio mixed bags from pairs of different combinations
            rng = np.random.default_rng(seed + 7); gen = torch.Generator().manual_seed(seed + 7)
            _, tr_c = eval_bags(split.train, n_fields, seed + 1, cap_train)[1:]
            m = int(len(tr_pos) * mix.get("ratio", 1.0)); ia = rng.integers(0, len(tr_pos), m); ib = rng.integers(0, len(tr_pos), m)
            keep = tr_c[ia] != tr_c[ib]; ia, ib = ia[keep], ib[keep]
            Hm = torch.cat([rep(mix_tiles(gather(Xtr, tr_pos[ia[i:i + 256]], dev), gather(Xtr, tr_pos[ib[i:i + 256]], dev), gen)).float().cpu()
                            for i in range(0, len(ia), 256)])
            H = torch.cat([H, Hm]); tr_y = np.concatenate([tr_y, np.maximum(tr_y[ia], tr_y[ib])])
        model = LinearProbe(H.mean(0), H.std(0) + 1e-6, K).to(dev)
        H, Ytr = H.to(dev), torch.from_numpy(tr_y).float().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-2), weight_decay=cfg.get("wd", 1e-2))
        for step in range(cfg.get("steps", 500)):
            loss = F.binary_cross_entropy_with_logits(model(H) + adj(Ytr, split.train.rows[tr_pos[:, 0]] if cov is not None else None), Ytr)
            opt.zero_grad(); loss.backward(); opt.step()
        score = lambda x: torch.sigmoid(model(rep(x)))
        n_params = sum(p.numel() for p in model.parameters())
    elif cfg["type"] == "set":
        D = Xtr.shape[-1]
        kw = cfg.get("model_kw", {})
        arch = {"deepsets": DeepSets, "popevidence": PopEvidence, "abmil": ClassAttentionMIL}[cfg["model"]]
        model = arch(D, K, **kw).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-2))
        sampler = BagSampler(split.train, n_fields, seed)
        mix_gen = torch.Generator().manual_seed(seed + 7)
        lam = cfg.get("lambda_cons", 0.0)
        lam_tile = cfg.get("lambda_tile", 0.0)
        steps = cfg.get("steps", 2000)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        model.train()
        for step in range(steps):
            pos, y, cid = sampler.sample(cfg.get("batch", 64))
            x = gather(Xtr, pos, dev)
            mix = cfg.get("mix")
            if mix:
                # CompMix on a fraction p of the batch: pair each bag with a bag of another combination
                pos_b, y_b, cid_b = sampler.sample(len(pos))
                sel = (np.random.default_rng(seed * 100003 + step).random(len(pos)) < mix.get("p", 0.5)) & (cid != cid_b)
                if sel.any():
                    s_ = torch.from_numpy(sel).to(dev)
                    xm = mix_tiles(x, gather(Xtr, pos_b, dev), mix_gen)
                    x = torch.where(s_.view(-1, 1, 1), xm, x)
                    y = np.where(sel[:, None], np.maximum(y, y_b), y); cid = np.where(sel, -1 - np.arange(len(cid)), cid)
            e, a = model(x)
            y = torch.from_numpy(y).float().to(dev)
            # CLA: the label-conditional prior is added in training only; scoring below uses e alone
            rows_b = split.train.rows[pos[:, 0]] if cov is not None else None
            loss = cfg.get("lambda_bce", 1.0) * F.binary_cross_entropy_with_logits(e + adj(y, rows_b), y)
            if lam_tile > 0:
                loss = loss + lam_tile * tile_evidence_loss(a, y, cfg.get("tile_q", 0.9), cfg.get("tile_pure", True))
            if lam > 0:
                loss = loss + lam * context_consistency(e, y, torch.from_numpy(cid).to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (step + 1) % max(steps // 4, 1) == 0:
                log(f"    step {step + 1}/{steps} loss {loss.item():.4f}")
        model.eval()
        score = lambda x: torch.sigmoid(model(x)[0])
        n_params = sum(p.numel() for p in model.parameters())

    S_ca = apply_chunked(score, Xca, ca_pos, dev).numpy()
    S_ev = apply_chunked(score, Xev, ev_pos, dev).numpy()
    A_ev = None
    if cfg["type"] == "set" and cfg.get("model") in ("popevidence", "abmil") and cfg.get("save_tile_evidence"):
        # first evidence channel per tile and species, (M, n*T, K), for tile-level ground-truth checks
        A_ev = apply_chunked(lambda x: model(x)[1][..., 0], Xev, ev_pos, dev).half().numpy()
    thr = argmax_f1_thresholds(S_ca, ca_y)
    return {"S": S_ev, "Y": ev_y, "C": ev_c, "A": A_ev, "pos": ev_pos, "rows": split.eval.rows, "P": (S_ev > thr).astype(np.int8), "thr": thr,
            "S_calib": S_ca, "Y_calib": ca_y, "n_params": int(n_params)}
