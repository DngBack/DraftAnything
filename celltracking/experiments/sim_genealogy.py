"""Zero-shot snapshot genealogy: learn "same ancestor H minutes ago" from
rendered simulated colonies only, evaluate on real OIAM phase contrast.

Each simulated frame is rescaled so a cell is ~12-19 px wide (OIAM-like) and
rendered as dark capsules on a bright background with a bright halo, blur,
per-cell intensity jitter, background gradient and noise, all randomised per
frame. Patches are normalised per sample, so absolute image statistics do not
matter. Same pair features, head and cumulative tau targets as
snapshot_genealogy.py; no real lineage label is used in training.

  python experiments/sim_genealogy.py train
  python experiments/sim_genealogy.py eval
"""

import argparse
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.family import family_edges  # noqa: E402
from experiments.colony_sim import SimMovie  # noqa: E402
from experiments.snapshot_genealogy import (CKPT, THRESHOLDS, Genealogy, Net,  # noqa: E402
                                            average_precision, patches)

SIM = Path(__file__).parents[1] / "outputs/sim/v1"


def lineage_factor(tracks, parent, rng, sigma):
    """Per-cell intensity factor inherited down the lineage (daughter = mother x noise)."""
    memo = {}

    def get(t):
        if t not in memo:
            p = parent.get(t, 0)
            base = get(p) if p else 1.0
            memo[t] = base * math.exp(np.random.default_rng((t * 7919) ^ salt).normal(0, sigma))
        return memo[t]
    salt = int(rng.integers(1 << 30))
    return np.array([get(int(t)) for t in tracks])


def render(f, shape, rng, parent=None):
    """Phase-contrast-like image of one frame (positions already in render px).

    With `parent` (render v2): per-cell intensity is inherited along the lineage,
    and the longest cells (about to divide) show a constriction at mid-cell."""
    h, w = shape
    sdf = np.full(shape, 1e3, np.float32)
    owner = -np.ones(shape, np.int32)
    pinch = np.zeros(len(f["pos"]), bool)
    if parent is not None and len(f["pos"]):
        pinch = f["major"] > np.quantile(f["major"], 1 - rng.uniform(0.05, 0.25))
    depth = rng.uniform(0.15, 0.45)
    for k, (c, u, ma, mi) in enumerate(zip(f["pos"], f["axis"], f["major"], f["minor"])):
        r = 2 * mi
        half = max(ma * math.sqrt(12) / 2 - r, 0)
        u = u / max(np.linalg.norm(u), 1e-6)
        ext = half + r + 6
        y0, y1 = int(max(c[0] - ext, 0)), int(min(c[0] + ext + 1, h))
        x0, x1 = int(max(c[1] - ext, 0)), int(min(c[1] + ext + 1, w))
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dy, dx = yy - c[0], xx - c[1]
        along = dy * u[0] + dx * u[1]
        t = np.clip(along, -half, half)
        d = np.sqrt((dy - t * u[0]) ** 2 + (dx - t * u[1]) ** 2) - r
        if pinch[k]:  # septum: narrow the rod around mid-cell
            d = d + depth * r * np.exp(-(along / (0.6 * r)) ** 2)
        sub = sdf[y0:y1, x0:x1]
        m = d < sub
        sub[m] = d[m]
        owner[y0:y1, x0:x1][m] = k
    bg, cell = rng.uniform(0.6, 1.0), rng.uniform(0.0, 0.3)
    halo, hw = rng.uniform(0.0, 0.6), rng.uniform(1.0, 4.0)
    edge = rng.uniform(0.5, 1.5)
    jitter = 1 + rng.normal(0, rng.uniform(0, 0.15), len(f["pos"]) + 1)
    if parent is not None:
        jitter[:-1] = lineage_factor(f["track"], parent, rng, rng.uniform(0.03, 0.12))
    inside = 1 / (1 + np.exp(sdf / edge))
    level = cell * jitter[owner]
    im = bg + (level - bg) * inside + halo * np.exp(-np.maximum(sdf, 0) / hw) * (1 - inside)
    gy, gx = np.mgrid[0:h, 0:w] / max(h, w)
    im = im + rng.normal(0, 0.15) * gy + rng.normal(0, 0.15) * gx
    im = ndimage.gaussian_filter(im, rng.uniform(0.7, 2.0))
    im = im + rng.normal(0, rng.uniform(0.01, 0.08), im.shape)
    return im.astype(np.float32)


class SimGenealogy(Genealogy):
    """Same interface as Genealogy (frames, img, edges, feat, tau, m) for one colony."""

    def __init__(self, path, n_frames, seed, v2=False):
        rng = np.random.default_rng(seed)
        self.m = SimMovie(path)
        scale = rng.uniform(12, 19) / (4 * np.median(np.concatenate(
            [f["minor"] for f in self.m.frames.values() if len(f["minor"])])))
        ok = [t for t, f in self.m.frames.items() if 30 <= len(f["track"]) <= 1500 and t >= 200]
        pick = sorted(rng.choice(ok, min(n_frames, len(ok)), replace=False)) if ok else []
        shape = tuple(int(s * scale) + 1 for s in self.m.image_shape)
        self.frames, self.img, self.edges, self.feat, self.tau = [], {}, {}, {}, {}
        for t in pick:
            f = dict(self.m.frames[t])
            f = {**f, "pos": f["pos"] * scale, "major": f["major"] * scale, "minor": f["minor"] * scale}
            self.m.frames[t] = f
            df = pd.DataFrame({"center_y": f["pos"][:, 0], "center_x": f["pos"][:, 1],
                               "axis_y": f["axis"][:, 0], "axis_x": f["axis"][:, 1],
                               "major_sd": f["major"], "minor_sd": f["minor"]})
            e, x = family_edges(df, 8)
            self.frames.append(t)
            self.edges[t], self.feat[t] = e, x
            self.tau[t] = np.array([self.split_time(int(f["track"][a]), int(f["track"][b]), t)
                                    for a, b in e], dtype=np.float32)
            self.img[t] = render(f, shape, rng, self.m.parent if v2 else None)
        self.m.frames = {t: self.m.frames[t] for t in self.frames}  # drop the rest to save memory


def _load(args):
    return SimGenealogy(*args)


def tag(a):
    t = "sim_genealogy_v2" if a.render_v2 else "sim_genealogy"
    return t if a.sim == "v1" else f"{t}_{a.sim}"


def norm_patches(img, pos, edges, dev, flip=None):
    p = patches(img, pos, edges, dev, flip)
    c = p[:, :1]
    c = (c - c.mean((2, 3), keepdim=True)) / (c.std((2, 3), keepdim=True) + 1e-4)
    return torch.cat([c, p[:, 1:]], 1)


def batch(data, rng, dev, n_frames=4, per_frame=256):
    xs, ims, ys = [], [], []
    for _ in range(n_frames):
        g = data[rng.integers(len(data))]
        t = g.frames[rng.integers(len(g.frames))]
        e = g.edges[t]
        k = rng.choice(len(e), min(per_frame, len(e)), replace=False)
        ek = e[k]
        if rng.random() < 0.5:
            ek = ek[:, ::-1].copy()
        flip = torch.as_tensor(rng.choice([-1.0, 1.0], len(ek)), device=dev, dtype=torch.float32)
        ims.append(norm_patches(g.img[t], g.m.frames[t]["pos"], ek, dev, flip))
        xs.append(torch.as_tensor(g.feat[t][k], device=dev))
        ys.append(torch.as_tensor(g.tau[t][k][:, None] < np.array(THRESHOLDS)[None], device=dev).float())
    return torch.cat(xs), torch.cat(ims), torch.cat(ys)


def train(a):
    dev = "cuda"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    paths = sorted((SIM.parent / a.sim).glob("colony_*.pkl"))[:a.n_colonies]
    t0 = time.time()
    with Pool(a.procs) as pool:
        data = [g for g in pool.map(_load, [(q, a.frames_per, i, a.render_v2) for i, q in enumerate(paths)]) if g.frames]
    print("loaded", sum(len(g.frames) for g in data), "sim frames", f"{time.time()-t0:.0f}s", flush=True)
    xg = np.concatenate([g.feat[t] for g in data for t in g.frames])
    mean, std = xg.mean(0), xg.std(0) + 1e-4
    mt, st = torch.as_tensor(mean, device=dev), torch.as_tensor(std, device=dev)
    models = {"geometry": Net(False).to(dev), "image": Net(True).to(dev)}
    opts = {k: torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4) for k, m in models.items()}
    scheds = {k: torch.optim.lr_scheduler.OneCycleLR(o, 1e-3, total_steps=a.steps) for k, o in opts.items()}
    for step in range(a.steps):
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
               CKPT / f"{tag(a)}.pt")
    np.save(CKPT / f"{tag(a)}_example.npy", data[0].img[data[0].frames[-1]])


@torch.no_grad()
def evaluate(a):
    import glob
    import json
    dev = "cuda"
    ck = torch.load(CKPT / f"{tag(a)}.pt", map_location=dev, weights_only=False)
    models = {}
    for k in ("geometry", "image"):
        models[k] = Net(k == "image").to(dev)
        models[k].load_state_dict(ck[k])
        models[k].eval()
    mt, st = torch.as_tensor(ck["mean"], device=dev), torch.as_tensor(ck["std"], device=dev)
    ws = [json.loads(line) for f in glob.glob(str(Path(__file__).parents[1] /
                                                  "outputs/benchmarks/v1/inputs/windows/*.jsonl"))
          for line in open(f)]
    rows = []
    for alias in ("M3", "M4"):
        need = {(w["target_frame"], w["horizon_frames"]) for w in ws
                if w["sequence_id"] == alias and w["schedule_id"] == "ends" and w["horizon_frames"] in (33, 66, 132)}
        g = Genealogy(alias, sorted({t for t, _ in need}))
        for t in g.frames:
            e, pos = g.edges[t], g.m.frames[t]["pos"]
            sc = {k: [] for k in models}
            for s in range(0, len(e), 512):
                ek = e[s:s + 512]
                x = (torch.as_tensor(g.feat[t][s:s + 512], device=dev) - mt) / st
                p1, p2 = norm_patches(g.img[t], pos, ek, dev), norm_patches(g.img[t], pos, ek[:, ::-1].copy(), dev)
                for k, m in models.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = 0.5 * (m(x, p1).float() + m(x, p2).float())
                    sc[k].append(torch.sigmoid(z).cpu().numpy())
            for h in (33, 66, 132):
                if (t, h) not in need:
                    continue
                y = g.tau[t] < h
                k = THRESHOLDS.index(h)
                for name, p in sc.items():
                    rows.append({"sequence": alias, "frame": t, "H": h, "model": f"{'simv2' if a.render_v2 else 'sim'}{'' if a.sim == 'v1' else '_' + a.sim}_{name}",
                                 "n_cells": len(g.m.frames[t]["track"]), "pos_rate": float(y.mean()),
                                 "ap": average_precision(np.concatenate(p)[:, k], y)})
        print(alias, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(CKPT.parent / f"{tag(a)}_eval.parquet", index=False)
    df["dense"] = df.n_cells > 600
    print(df.groupby(["H", "model"])[["ap", "pos_rate"]].mean().round(4).to_string())
    print("dense", df[df.dense].groupby(["H", "model"])[["ap", "pos_rate"]].mean().round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "render"])
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--n-colonies", type=int, default=200)
    p.add_argument("--frames-per", type=int, default=6)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--render-v2", action="store_true", help="lineage-inherited intensity + septum constriction")
    p.add_argument("--sim", default="v1", help="colony set under outputs/sim/")
    a = p.parse_args()
    if a.mode == "render":  # quick visual check of one rendered frame
        g = SimGenealogy(sorted((SIM.parent / a.sim).glob("colony_*.pkl"))[3], 1, 0, a.render_v2)
        np.save(CKPT / f"{tag(a)}_example.npy", g.img[g.frames[0]])
        print(g.img[g.frames[0]].shape, len(g.edges[g.frames[0]]), float((g.tau[g.frames[0]] < 33).mean()))
        return
    (train if a.mode == "train" else evaluate)(a)


if __name__ == "__main__":
    main()
