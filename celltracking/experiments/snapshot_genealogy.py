"""Genealogy from a single snapshot: is family membership visible in the image?

For neighbouring cells (kNN, k=8) in ONE frame, predict tau = minutes since
their lineages split (time back to the last common division), as cumulative
binary targets tau < {8,16,33,66,132,264}. Same ancestor H minutes ago <=>
tau < H, so one model serves every horizon (an ultrametric over the frame).

Two models, same data and head: geometry-only (10 single-frame features from
cellanc.family) and image+geometry (pair-aligned 64x64 phase-contrast patch).
Train on M0-M2, report AP on the benchmark target frames of M3/M4.

  python experiments/snapshot_genealogy.py train
  python experiments/snapshot_genealogy.py eval
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.family import FEATURE_NAMES, family_edges  # noqa: E402
from experiments.family_set import RESULTS, SEQUENCE  # noqa: E402
from experiments.lineage_glue import INDEX, Movie  # noqa: E402

THRESHOLDS = (8, 16, 33, 66, 132, 264)
HALF = 32  # patch half-size in pixels
CKPT = RESULTS / "glue"


class Genealogy:
    """Single-frame cells + per-pair split time, for one OIAM movie."""

    def __init__(self, alias, frames, pseudo=False):
        self.m = Movie(alias)
        pl = None
        if pseudo:  # annotation-free: split times from chained one-frame links (chained_pseudo_lineage)
            from experiments.chained_pseudo_lineage import PseudoMovie
            pl = PseudoMovie(alias).pl
        paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
        self.frames = [t for t in frames if t in self.m.frames and len(self.m.frames[t]["track"]) >= 8]
        self.img, self.edges, self.feat, self.tau = {}, {}, {}, {}
        for t in self.frames:
            f = self.m.frames[t]
            df = pd.DataFrame({"center_y": f["pos"][:, 0], "center_x": f["pos"][:, 1],
                               "axis_y": f["axis"][:, 0], "axis_x": f["axis"][:, 1],
                               "major_sd": f["major"], "minor_sd": f["minor"]})
            e, x = family_edges(df, 8)
            self.edges[t], self.feat[t] = e, x
            if pl is None:
                self.tau[t] = np.array([self.split_time(int(f["track"][a]), int(f["track"][b]), t)
                                        for a, b in e], dtype=np.float32)
            else:
                self.tau[t] = self.pseudo_tau(pl, e, t)
            im = tifffile.imread(paths[t]).astype(np.float32)
            self.img[t] = (im - im.mean()) / (im.std() + 1e-6)

    def pseudo_tau(self, pl, e, t):
        """Smallest threshold H at which both cells chain back to the same pseudo-ancestor (minus 0.5),
        so tau < H reproduces the pseudo labels; 1e4 if they never share one."""
        tau = np.full(len(e), 1e4, np.float32)
        for h in sorted(THRESHOLDS, reverse=True):
            if t - h not in self.m.frames:
                continue
            anc = pl.ancestors(t - h, t)
            share = (anc[e[:, 0]] == anc[e[:, 1]]) & (anc[e[:, 0]] >= 0)
            tau[share] = h - 0.5
        return tau

    def chain(self, track):
        out = []
        while track:
            out.append(track)
            track = self.m.parent.get(track, 0)
        return out

    def split_time(self, ti, tj, b):
        ci, cj = self.chain(ti), set(self.chain(tj))
        for k, tr in enumerate(ci):
            if tr in cj:
                if k == 0:
                    return 0.0
                return float(b - self.m.start[ci[k - 1]])
        return 1e4


def patches(img, pos, edges, dev, flip=None):
    """Pair-aligned crops: x axis along i->j, centred at the midpoint."""
    im = torch.as_tensor(img, device=dev)[None, None]
    h, w = img.shape
    p = torch.as_tensor(pos, device=dev, dtype=torch.float32)
    a, b = p[edges[:, 0]], p[edges[:, 1]]
    mid = (a + b) / 2
    u = F.normalize(b - a, dim=1)
    v = torch.stack([-u[:, 1], u[:, 0]], 1)
    if flip is not None:
        v = v * flip[:, None]
    g = torch.linspace(-1, 1, 2 * HALF, device=dev)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    pts = mid[:, None, None] + HALF * (gx[..., None] * u[:, None, None] + gy[..., None] * v[:, None, None])
    grid = torch.stack([pts[..., 1] / (w - 1) * 2 - 1, pts[..., 0] / (h - 1) * 2 - 1], -1)
    n = len(edges)  # sample every crop from the one image: no n-fold copy
    crop = F.grid_sample(im, grid.reshape(1, n * 2 * HALF, 2 * HALF, 2), align_corners=True)
    crop = crop.reshape(1, n, 2 * HALF, 2 * HALF).transpose(0, 1)
    d = (b - a).norm(dim=1) / 2 / HALF
    blob = torch.exp(-((gx[None] + d[:, None, None]) ** 2 + gy[None] ** 2) / 0.01) + \
        torch.exp(-((gx[None] - d[:, None, None]) ** 2 + gy[None] ** 2) / 0.01)
    return torch.cat([crop, blob[:, None]], 1)


class Net(nn.Module):
    def __init__(self, image=True):
        super().__init__()
        self.image = image
        if image:
            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                     nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
            self.cnn = nn.Sequential(blk(2, 32), nn.MaxPool2d(2), blk(32, 64), nn.MaxPool2d(2),
                                     blk(64, 128), nn.MaxPool2d(2), blk(128, 256),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.geo = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + (256 if image else 0), 256), nn.ReLU(),
                                  nn.Linear(256, len(THRESHOLDS)))

    def forward(self, x_geo, x_img=None):
        z = self.geo(x_geo)
        if self.image:
            z = torch.cat([z, self.cnn(x_img)], 1)
        return self.head(z)


def batch(data, rng, dev, n_frames=4, per_frame=256, train=True):
    xs, ims, ys = [], [], []
    for _ in range(n_frames):
        g = data[rng.integers(len(data))]
        t = g.frames[rng.integers(len(g.frames))]
        e = g.edges[t]
        k = rng.choice(len(e), min(per_frame, len(e)), replace=False)
        ek = e[k]
        if train and rng.random() < 0.5:
            ek = ek[:, ::-1].copy()
        flip = torch.as_tensor(rng.choice([-1.0, 1.0], len(ek)), device=dev, dtype=torch.float32)
        ims.append(patches(g.img[t], g.m.frames[t]["pos"], ek, dev, flip))
        xs.append(torch.as_tensor(g.feat[t][k], device=dev))
        ys.append(torch.as_tensor(g.tau[t][k][:, None] < np.array(THRESHOLDS)[None], device=dev).float())
    return torch.cat(xs), torch.cat(ims), torch.cat(ys)


def train(args):
    dev = "cuda"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    data = [Genealogy(a, range(250, 800, 2), pseudo=args.pseudo) for a in ("M0", "M1", "M2")]
    print("loaded", sum(len(g.frames) for g in data), "frames", flush=True)
    xg = np.concatenate([g.feat[t] for g in data for t in g.frames])
    mean, std = xg.mean(0), xg.std(0) + 1e-4
    mt, st = torch.as_tensor(mean, device=dev), torch.as_tensor(std, device=dev)
    models = {"geometry": Net(False).to(dev), "image": Net(True).to(dev)}
    opts = {k: torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4) for k, m in models.items()}
    scheds = {k: torch.optim.lr_scheduler.OneCycleLR(o, 1e-3, total_steps=args.steps)
              for k, o in opts.items()}
    t0 = time.time()
    for step in range(args.steps):
        x, im, y = batch(data, rng, dev)
        x = (x - mt) / st
        losses = {}
        for k, m in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m(x, im)
            loss = F.binary_cross_entropy_with_logits(z.float(), y)
            opts[k].zero_grad()
            loss.backward()
            opts[k].step()
            scheds[k].step()
            losses[k] = loss.item()
        if (step + 1) % 500 == 0:
            print(step + 1, {k: round(v, 4) for k, v in losses.items()}, f"{time.time()-t0:.0f}s", flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({k: m.state_dict() for k, m in models.items()} | {"mean": mean, "std": std},
               CKPT / f"snapshot_genealogy{'_pseudo' if args.pseudo else ''}.pt")


def average_precision(score, y):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-score)
    y = y[o]
    return float((np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / y.sum())


@torch.no_grad()
def score_frame(models, g, t, mt, st, dev):
    e = g.edges[t]
    out = {}
    for k, m in models.items():
        zs = []
        for s in range(0, len(e), 512):
            ek = e[s:s + 512]
            x = (torch.as_tensor(g.feat[t][s:s + 512], device=dev) - mt) / st
            im = patches(g.img[t], g.m.frames[t]["pos"], ek, dev)
            im2 = patches(g.img[t], g.m.frames[t]["pos"], ek[:, ::-1].copy(), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = 0.5 * (m(x, im).float() + m(x, im2).float())
            zs.append(torch.sigmoid(z).cpu().numpy())
        out[k] = np.concatenate(zs)
    return out


def evaluate(args):
    import json
    import glob
    dev = "cuda"
    sfx = "_pseudo" if args.pseudo else ""
    ck = torch.load(CKPT / f"snapshot_genealogy{sfx}.pt", map_location=dev, weights_only=False)
    models = {}
    for k in ("geometry", "image"):
        models[k] = Net(k == "image").to(dev)
        models[k].load_state_dict(ck[k])
        models[k].eval()
    mt, st = torch.as_tensor(ck["mean"], device=dev), torch.as_tensor(ck["std"], device=dev)
    ws = [json.loads(l) for f in glob.glob(str(Path(__file__).parents[1] / "outputs/benchmarks/v1/inputs/windows/*.jsonl"))
          for l in open(f)]
    rows = []
    for alias in ("M3", "M4"):
        need = {(w["target_frame"], w["horizon_frames"]) for w in ws
                if w["sequence_id"] == alias and w["schedule_id"] == "ends" and w["horizon_frames"] in (33, 66, 132)}
        g = Genealogy(alias, sorted({t for t, _ in need}))
        for t in g.frames:
            sc = score_frame(models, g, t, mt, st, dev)
            for h in (33, 66, 132):
                if (t, h) not in need:
                    continue
                y = g.tau[t] < h
                k = THRESHOLDS.index(h)
                for name, p in sc.items():
                    rows.append({"sequence": alias, "frame": t, "H": h, "model": name,
                                 "n_cells": len(g.m.frames[t]["track"]), "pos_rate": float(y.mean()),
                                 "ap": average_precision(p[:, k], y)})
        print(alias, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"snapshot_genealogy{sfx}_eval.parquet", index=False)
    df["dense"] = df.n_cells > 600
    print(df.groupby(["H", "model"])[["ap", "pos_rate"]].mean().round(4).to_string())
    print("dense", df[df.dense].groupby(["H", "model"])[["ap", "pos_rate"]].mean().round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--pseudo", action="store_true", help="train on chained pseudo-lineage (no annotation); eval scores it vs GT")
    a = p.parse_args()
    (train if a.mode == "train" else evaluate)(a)


if __name__ == "__main__":
    main()
