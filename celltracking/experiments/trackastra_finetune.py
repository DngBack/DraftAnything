"""Trackastra general_2d fine-tuned on OIAM per gap (docs/trackastra_finetune_protocol.md).

Data: M0/M1 subsampled at stride H with every start frame s (all offsets 0..H-1),
windows of `window` (=4, from the pretrained config) subsampled frames
s, s+H, s+2H, s+3H with timepoints 0..3. Each detection is a one-frame tracklet whose
parent is its GT ancestor at the previous subsampled frame, so the association target
links each detection to all its ancestors and descendants inside the window (as
_ctc_assoc_matrix does). Features: wrfeat (regionprops2) from GT masks and per-frame
normalised images, exactly as CTCData._load_wrfeat. Sampling, cropping, augmentation,
max_tokens clipping and random coordinate offset follow CTCData._getitem_wrfeat and
BalancedBatchSampler (weight_by_ndivs) with the pretrained train_config
(crop 320, augment 3, max_tokens 2560, n_pool 8). Loss: WrappedLightningModule._common_step
of scripts/train.py (weigertlab/trackastra), with causal_norm=quiet_softmax,
delta_cutoff=1, div_upweight=2. AdamW(lr 1e-4, wd 1e-5), fp16 autocast + GradScaler,
linear warmup (1k steps from 1e-2) then cosine to 1e-3 over 20k steps.
Validation every 1k steps on M2 windows (argmax decoding, mean per-window accuracy);
the best fine-tuned checkpoint (step >= 1k) is kept.

  python experiments/trackastra_finetune.py train --horizon 33 [--steps 20000 --hours 3]
  python experiments/trackastra_finetune.py eval            # M3/M4 once -> trackastra_ft.parquet
"""

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.fast_microcolony_external_test import family_f1  # noqa: E402
from experiments.lineage_glue import INDEX  # noqa: E402

CKPT = ROOT / "outputs/checkpoints/trackastra_ft"
CACHE = ROOT / "outputs/cache/trackastra_ft"
EPS = torch.finfo(torch.float32).eps


# ----------------------------------------------------------------------------- data
def frame_features(alias):
    """Per-frame WRFeatures (t_start=0) for the whole movie, cached."""
    from trackastra.data.wrfeat import WRFeatures
    from trackastra.utils import normalize
    path = CACHE / f"{alias}_feats.pkl"
    if path.exists():
        return joblib.load(path)
    fr = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index")

    def one(t):
        img = normalize(tifffile.imread(fr.image_path[t]).astype(np.float32))
        mask = tifffile.imread(fr.mask_path[t])
        return WRFeatures.from_mask_img(mask=mask[None], img=img[None], t_start=0)

    feats = joblib.Parallel(n_jobs=16)(joblib.delayed(one)(t) for t in fr.index)
    feats = dict(zip(fr.index.astype(int), feats))
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(feats, path)
    return feats


class Lineage:
    def __init__(self, alias):
        tr = pd.read_parquet(INDEX / SEQUENCE[alias] / "tracks.parquet")
        self.start = dict(zip(tr.gt_track_id.astype(int), tr.start_frame.astype(int)))
        self.parent = dict(zip(tr.gt_track_id.astype(int), tr.parent_id.astype(int)))
        self.anc = lru_cache(maxsize=None)(self._anc)

    def _anc(self, track, frame, at=None):  # = lineage_glue.Movie.ancestor (at: frame of `track`, unused)
        while self.start.get(track, -1) > frame:
            track = self.parent.get(track, 0)
            if track == 0:
                return 0
        return track


class PseudoTrackLineage:
    """Annotation-free stand-in for Lineage (docs/trackastra_pseudo_protocol.md): ancestry from chained
    one-frame links (chained_pseudo_lineage.PseudoMovie, div_cost 10, as for the self-labelled LineageRAFT).
    Mask labels are used only as per-frame detection ids; anc(label, f, at) = label at frame f of the
    pseudo-ancestor of detection `label` at frame `at` (0 = none)."""

    def __init__(self, alias):
        from experiments.chained_pseudo_lineage import PseudoMovie
        pm = PseudoMovie(alias)
        self.links = pm.pl.links
        self.tracks = {t: f["track"].astype(np.int64) for t, f in pm.frames.items()}
        self.row = {t: {int(k): i for i, k in enumerate(tr)} for t, tr in self.tracks.items()}
        self.chain = lru_cache(maxsize=8192)(self._chain)

    def _chain(self, f, at):  # pseudo-ancestor index in frame f of every cell of frame `at`
        idx = np.arange(len(self.tracks[at]))
        for t in range(at, f, -1):
            idx = np.where(idx >= 0, self.links[t][np.maximum(idx, 0)], -1)
        return idx

    def anc(self, track, frame, at):
        i = self.row.get(at, {}).get(track)
        if i is None:
            return 0
        if frame >= at:
            return track
        j = self.chain(frame, at)[i]
        return int(self.tracks[frame][j]) if j >= 0 else 0


def assoc(labels, timepoints, frames, lin):
    """A[i, j] = detections i, j are the same cell or ancestor/descendant (CTC family)."""
    w = len(frames)
    am = np.full((len(labels), w), -2, dtype=np.int64)
    for j, (lb, t) in enumerate(zip(labels, timepoints)):
        for k in range(t + 1):
            am[j, k] = lin.anc(int(lb), int(frames[k]), int(frames[t]))
    up = am[np.arange(len(labels))[None, :], timepoints[:, None]] == labels[:, None]  # [i,j]: anc(j, t_i) == i
    return up | up.T


class GapData(torch.utils.data.Dataset):
    def __init__(self, aliases, horizon, cfg):
        from trackastra.data.data import CTCData
        self.h, self.w = horizon, cfg["window"]
        self.max_tokens = cfg["max_tokens"]
        _, self.augmenter, self.cropper = CTCData._setup_features_augs_wrfeat(
            None, cfg["ndim"], cfg["features"], cfg["augment"], cfg["crop_size"])
        if cfg.get("aug_mode") == "norot":  # docs/trackastra_norot_protocol.md: augment=3 without flip/rotation/shear
            from trackastra.data import wrfeat
            self.augmenter = wrfeat.WRAugmentationPipeline([
                wrfeat.WRRandomAffine(p=0.8, degrees=0, scale=(0.5, 2), shear=(0, 0)),
                wrfeat.WRRandomBrightness(p=0.8),
                wrfeat.WRRandomMovement(offset=(-10, 10), p=0.3),
                wrfeat.WRRandomOffset(p=0.8, offset=(-3, 3)),
            ])
        self.feats = {a: frame_features(a) for a in aliases}
        self.lin = {a: (PseudoTrackLineage if cfg.get("pseudo") else Lineage)(a) for a in aliases}
        self.items, self.n_objects, self.n_divs = [], [], []
        for h in (horizon if isinstance(horizon, (list, tuple)) else [horizon]):  # list: docs/matched_trackastra_protocol.md
            for a in aliases:
                last = max(self.feats[a])
                for s in range(0, last - (self.w - 1) * h + 1):
                    frames = [s + k * h for k in range(self.w)]
                    if any(len(self.feats[a][f]) == 0 for f in frames):
                        continue
                    self.items.append((a, frames))
                    self.n_objects.append(sum(len(self.feats[a][f]) for f in frames))
                    self.n_divs.append(self._ndivs(a, frames))
            print(f"H={h}: {len(self.items)} windows so far", flush=True)
        self.n_divs = np.array(self.n_divs)
        print(f"H={horizon}: {len(self.items)} windows, objects/window median "
              f"{int(np.median(self.n_objects))}, ndivs median {np.median(self.n_divs)}", flush=True)

    def _ndivs(self, a, frames):
        """= CTCData._get_ndivs: #columns whose max blockwise row-sum of A is exactly 2."""
        lin, f = self.lin[a], self.feats[a]
        n = 0
        for k, fk in enumerate(frames[:-1]):
            lk = f[fk].labels
            best = np.ones(len(lk), dtype=int)
            for fm in frames[k + 1:]:
                anc = np.array([lin.anc(int(b), fk, fm) for b in f[fm].labels])
                cnt = (anc[None, :] == lk[:, None]).sum(1)
                best = np.maximum(best, cnt)
            n += int((best == 2).sum())
        return n

    def window_feat(self, a, frames):
        from trackastra.data.wrfeat import WRFeatures
        parts = []
        for k, fr in enumerate(frames):
            x = self.feats[a][fr]
            parts.append(WRFeatures(coords=x.coords, labels=x.labels,
                                    timepoints=np.full(len(x.labels), k, dtype=np.int32),
                                    features=x.features))
        return WRFeatures.concat(parts)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):  # = CTCData._getitem_wrfeat, with A computed for the kept tokens
        a, frames = self.items[n]
        feat = self.window_feat(a, frames)
        if self.cropper is not None:
            cropped_feat, cropped_idx = self.cropper(feat)
            if len(np.unique(feat.timepoints[cropped_idx])) == self.w:
                feat = cropped_feat
        if self.augmenter is not None:
            feat = self.augmenter(feat)
        coords0 = torch.from_numpy(np.concatenate((feat.timepoints[:, None], feat.coords), -1)).float()
        features = torch.from_numpy(feat.features_stacked).float()
        labels = feat.labels
        timepoints = feat.timepoints
        if self.max_tokens and len(timepoints) > self.max_tokens:
            time_incs = np.where(timepoints - np.roll(timepoints, 1))[0]
            n_elems = time_incs[np.searchsorted(time_incs, self.max_tokens) - 1]
            timepoints, labels = timepoints[:n_elems], labels[:n_elems]
            coords0, features = coords0[:n_elems], features[:n_elems]
        A = assoc(labels.astype(np.int64), timepoints.astype(np.int64), frames, self.lin[a])
        coords = coords0.clone()
        if self.augmenter is not None:
            coords[:, 1:] += torch.randint(0, 512, (1, coords.shape[1] - 1))
        return dict(features=features, coords0=coords0, coords=coords,
                    assoc_matrix=torch.from_numpy(A.astype(np.float32)),
                    timepoints=torch.from_numpy(timepoints).long(),
                    labels=torch.from_numpy(labels).long())


class BalancedBatches(torch.utils.data.Sampler):
    """= trackastra BalancedBatchSampler.sample_batches (weight_by_ndivs), endless."""

    def __init__(self, ds, batch_size, n_pool, seed):
        self.ds, self.bs, self.n_pool = ds, batch_size, n_pool
        p = 1 + np.sqrt(ds.n_divs)
        self.p = p / p.sum()
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        no = self.ds.n_objects
        while True:
            idx = self.rng.choice(len(self.ds), self.n_pool * self.bs, replace=True, p=self.p)
            idx = sorted(idx, key=lambda i: no[i])
            jj = np.arange(0, len(idx), self.bs)
            self.rng.shuffle(jj)
            for j in jj:
                yield [int(i) for i in idx[j:j + self.bs]]


# ----------------------------------------------------------------------------- loss
def common_step(model, batch, causal_norm, delta_cutoff, div_upweight):
    """= WrappedLightningModule._common_step of trackastra scripts/train.py."""
    from trackastra.utils import blockwise_causal_norm, blockwise_sum
    feats, coords, A = batch["features"], batch["coords"], batch["assoc_matrix"]
    timepoints, padding_mask = batch["timepoints"], batch["padding_mask"].bool()
    A_pred = model(coords, feats, padding_mask=padding_mask)
    A_pred = A_pred.clamp(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)
    mask_invalid = torch.logical_or(padding_mask.unsqueeze(1), padding_mask.unsqueeze(2))
    A_pred = A_pred.masked_fill(mask_invalid, 0)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(A_pred, A, reduction="none")
    if causal_norm != "none":
        A_pred_soft = torch.stack([blockwise_causal_norm(_A, _t, mode=causal_norm, mask_invalid=_m)
                                   for _A, _t, _m in zip(A_pred, timepoints, mask_invalid)])
        with torch.autocast("cuda", enabled=False):
            loss = 0.01 * loss.float() + torch.nn.functional.binary_cross_entropy(
                A_pred_soft.float(), A, reduction="none")
    with torch.no_grad():
        bs1 = torch.stack([blockwise_sum(a, t, dim=-1) for a, t in zip(A, timepoints)], 0)
        bs2 = torch.stack([blockwise_sum(a, t, dim=-2) for a, t in zip(A, timepoints)], 0)
        block_sum = A * (bs1 + bs2)
        loss_weight = 1 + 1.0 * (block_sum == 2) + div_upweight * (block_sum > 2)
    loss = loss * loss_weight
    dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
    mask = (torch.logical_and(dt > 0, dt <= delta_cutoff) * ~mask_invalid).float()
    loss_normalized = loss * mask / (mask.sum(dim=(1, 2), keepdim=True) + EPS)
    loss_per_sample = loss_normalized.sum(dim=(1, 2))
    prefactor = torch.pow(mask.sum(dim=(1, 2)), 0.2)
    return (loss_per_sample * prefactor / (prefactor.sum() + EPS)).sum()


# ----------------------------------------------------------------------------- scoring
def load_frames(alias, ta, tb):
    fr = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index")
    imgs = np.stack([tifffile.imread(fr.image_path[t]) for t in (ta, tb)])
    masks = np.stack([tifffile.imread(fr.mask_path[t]) for t in (ta, tb)])
    return imgs, masks


def argmax_links(pred):
    node = {n["id"]: (int(n["time"]), int(n["label"])) for n in pred["nodes"]}
    best = {}
    for (u, v), wt in pred["weights"]:
        (t0, l0), (t1, l1) = sorted([node[int(u)], node[int(v)]])
        if t0 == 0 and t1 == 1 and wt > best.get(l1, (-1, 0))[0]:
            best[l1] = (float(wt), l0)
    return {k: v[1] for k, v in best.items()}


def score(e, links):
    """Rows exactly as trackastra_gap_baseline.py; links: {name: {target label: anchor label}}."""
    w = e["window"]
    tg = e["targets"]
    ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
    anc, tgt = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
    ro = {int(v): k for k, v in enumerate(tgt.local_detection_id)}
    co = {int(v): k for k, v in enumerate(anc.local_detection_id)}
    ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
    if not len(ok):
        return []
    r = ok.target_detection_id.map(ro).to_numpy()
    truth = ok.ancestor_detection_id.map(co).to_numpy()
    by_track = {int(t): k for k, t in enumerate(anc.gt_track_id)}
    near = cKDTree(anc[["center_y", "center_x"]].to_numpy()).query(tgt[["center_y", "center_x"]].to_numpy())[1]
    labels = tgt.gt_track_id.astype(int).to_numpy()
    base = {"sequence": w["sequence_id"], "H": w["horizon_frames"], "window_id": w["window_id"],
            "n_anchors": len(co), "n_scored": len(r)}
    rows = []
    for name, link in links.items():
        g = np.array([by_track.get(link.get(lb, -1), near[k]) for k, lb in enumerate(labels)])[r]
        rows.append({**base, "method": name, "accuracy": float(np.mean(g == truth)),
                     "set_f1": family_f1(truth, g), "linked": float(np.mean([lb in link for lb in labels[r]]))})
    g = near[r]
    rows.append({**base, "method": "nearest", "accuracy": float(np.mean(g == truth)),
                 "set_f1": family_f1(truth, g), "linked": 1.0})
    return rows


class Validator:
    """M2 argmax accuracy; features cached, prediction = Trackastra._predict."""

    def __init__(self, alias, horizon, train_args, coord_dim, max_windows=None):
        from trackastra.data import get_features
        from trackastra.utils import normalize
        windows = Windows(BENCH, ROOT / "data").windows
        self.ex = examples(alias, windows, compensate=False, horizon=horizon)[:max_windows]
        self.feats = []
        for e in self.ex:
            imgs, masks = load_frames(alias, e["window"]["anchor_frame"], e["window"]["target_frame"])
            self.feats.append(get_features(detections=masks, imgs=normalize(imgs),
                                           features=train_args["features"], ndim=coord_dim,
                                           progbar_class=lambda x, **k: x))
            e["n_dims"] = masks.ndim - 1

    @torch.no_grad()
    def __call__(self, transformer, batch_size=4):
        from trackastra.data import build_windows
        from trackastra.model.predict import predict_windows
        transformer.eval()
        rows = []
        for e, f in zip(self.ex, self.feats):
            wins = build_windows(f, window_size=transformer.config["window"],
                                 progbar_class=lambda x, **k: x, as_torch=True)
            pred = predict_windows(windows=wins, features=f, model=transformer, edge_threshold=0.05,
                                   spatial_dim=e["n_dims"], progbar_class=lambda x, **k: x,
                                   batch_size=batch_size)
            rows += score(e, {"argmax": argmax_links(pred)})
        transformer.train()
        df = pd.DataFrame(rows)
        return {m: float(g.accuracy.mean()) for m, g in df.groupby("method")}


# ----------------------------------------------------------------------------- train / eval
def train(a):
    from trackastra.data import collate_sequence_padding
    from trackastra.model import Trackastra
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    dev = "cuda"
    model = Trackastra.from_pretrained("general_2d", device=dev)
    cfg = {**model.train_args, "aug_mode": a.aug, "pseudo": a.pseudo}
    net = model.transformer
    # --multi: one model over several horizons, best checkpoint kept per validated gap in h{H}{suffix}/
    gaps = a.val_horizons if a.multi else [a.horizon]
    outs = {h: CKPT / f"h{h}{a.suffix}" for h in gaps}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    # --no-val: M2 is training data, so no selection; the last checkpoint is saved as best.pt
    vals = {} if a.no_val else {h: Validator("M2", h, cfg, net.config["coord_dim"], a.val_windows) for h in gaps}
    ds = GapData(a.train, a.multi or a.horizon, cfg)
    tag = f"multi{a.suffix}" if a.multi else f"h{a.horizon}{a.suffix}"
    loader = iter(torch.utils.data.DataLoader(
        ds, batch_sampler=BalancedBatches(ds, a.batch_size, cfg["n_pool_sampler"], a.seed),
        num_workers=a.workers, collate_fn=collate_sequence_padding, pin_memory=True,
        persistent_workers=a.workers > 0, worker_init_fn=lambda i: np.random.seed(a.seed * 100 + i)))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)

    def factor(step):  # WarmupCosineLRScheduler, in steps
        if step < a.warmup:
            return 1e-2 + (1 - 1e-2) * step / a.warmup
        rel = (step - a.warmup) / (a.steps - a.warmup + 1)
        return 0.5 * (1 + math.cos(math.pi * rel)) * (1 - 1e-3) + 1e-3

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    scaler = torch.amp.GradScaler("cuda")
    hist = [{"step": 0, **{f"val_acc_{h}": v(net)["argmax"] for h, v in vals.items()}, "train_loss": None,
             "elapsed_h": 0.0}]
    print(json.dumps(hist[-1]), flush=True)
    best, t0, run = {h: -1.0 for h in gaps}, time.time(), []
    net.train()
    step = 0
    for step in range(1, a.steps + 1):
        batch = next(loader)
        batch = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = common_step(net, batch, cfg["causal_norm"], cfg["delta_cutoff"], cfg["div_upweight"])
        opt.zero_grad(set_to_none=True)
        if torch.isnan(loss):
            print("NaN loss, skipping", flush=True)
            continue
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run.append(float(loss.detach()))
        if step % 100 == 0:
            print(f"step {step} loss {np.mean(run[-100:]):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / 60:.1f} min, mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
                  flush=True)
        over = (time.time() - t0) / 3600 >= a.hours
        if vals and (step % a.val_every == 0 or over):
            accs = {h: v(net)["argmax"] for h, v in vals.items()}
            hist.append({"step": step, **{f"val_acc_{h}": acc for h, acc in accs.items()},
                         "train_loss": float(np.mean(run[-a.val_every:])), "elapsed_h": (time.time() - t0) / 3600})
            print(json.dumps(hist[-1]), flush=True)
            for h, acc in accs.items():
                if acc > best[h]:
                    best[h] = acc
                    torch.save({"model": net.state_dict(), "step": step, "val_acc": acc}, outs[h] / "best.pt")
                pd.DataFrame(hist).to_csv(outs[h] / "history.csv", index=False)
        if over:
            print("time budget reached", flush=True)
            break
    if a.no_val:
        for o in outs.values():
            torch.save({"model": net.state_dict(), "step": step, "val_acc": float("nan")}, o / "best.pt")
        hist.append({"step": step, "train_loss": float(np.mean(run[-a.val_every:])), "elapsed_h": (time.time() - t0) / 3600})
    for o in outs.values():
        pd.DataFrame(hist).to_csv(o / "history.csv", index=False)
    print("done", tag, "best", best, flush=True)


def evaluate(a):
    from trackastra.model import Trackastra
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for h in a.horizons:
        model = Trackastra.from_pretrained("general_2d", device=a.device)
        ck = torch.load(CKPT / f"h{h}{a.suffix}" / "best.pt", map_location=a.device)
        model.transformer.load_state_dict(ck["model"])
        model.transformer.eval()
        print(f"H={h}: checkpoint step {ck['step']} M2 val acc {ck['val_acc']:.4f}", flush=True)
        for alias in a.sequences:
            for e in examples(alias, windows, compensate=False, horizon=h):
                w = e["window"]
                imgs, masks = load_frames(alias, w["anchor_frame"], w["target_frame"])
                pred = model._predict(imgs, masks)
                graph, _ = model.track(imgs, masks, mode="greedy", progbar_class=lambda x, **k: x)
                greedy = {}
                for u, v in graph.edges:
                    du, dv = graph.nodes[u], graph.nodes[v]
                    if int(du["time"]) == 0 and int(dv["time"]) == 1:
                        greedy[int(dv["label"])] = int(du["label"])
                rows += score(e, {"trackastra_ft_greedy": greedy, "trackastra_ft_argmax": argmax_links(pred)})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"{a.out}.parquet", index=False)
    df["dense"] = df.n_anchors > 200
    print(df.pivot_table(index="method", columns="H", values="accuracy").round(4).to_string())
    print("dense\n", df[df.dense].pivot_table(index="method", columns="H", values="accuracy").round(4).to_string())
    print("set_f1\n", df.pivot_table(index="method", columns="H", values="set_f1").round(4).to_string())
    print("linked\n", df.pivot_table(index="method", columns="H", values="linked").round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--horizon", type=int, default=None)
    t.add_argument("--multi", nargs="+", type=int, default=None, help="train one model on these horizons")
    t.add_argument("--val-horizons", nargs="+", type=int, default=[33, 66, 132])
    t.add_argument("--train", nargs="+", default=["M0", "M1"])
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--hours", type=float, default=3.0)
    t.add_argument("--val-every", type=int, default=1000)
    t.add_argument("--val-windows", type=int, default=None)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--suffix", default="")
    t.add_argument("--aug", choices=["official", "norot"], default="official")
    t.add_argument("--no-val", action="store_true", help="no M2 validation; keep the last checkpoint")
    t.add_argument("--pseudo", action="store_true", help="train on chained pseudo-lineage (no annotation)")
    e = sub.add_parser("eval")
    e.add_argument("--sequences", nargs="+", default=["M3", "M4"])
    e.add_argument("--horizons", nargs="+", type=int, default=[33, 66, 132])
    e.add_argument("--device", default="cuda")
    e.add_argument("--suffix", default="")
    e.add_argument("--out", default="trackastra_ft")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
