"""LineageGlue: set-to-set transformer for sparse two-snapshot ancestry.

Train on *all* frame pairs (a, a+H) sampled from dense OIAM training movies
(GT lineage gives labels for free), evaluate on the fixed benchmark windows.
Inference input per frame: cell centers + single-frame mask second moments,
image size, and the elapsed time H. No temporal IDs or GT velocity.

Model: tokens for anchors (frame a) and targets (frame b), coordinates
canonicalised by a parameter-free radial expansion; self/cross attention; each
anchor predicts its descendant-family centre (a learned deformation) and the
target->anchor logits combine that centre with a learned similarity, plus a
dustbin for targets without a visible ancestor.

  python experiments/lineage_glue.py train --train M0 M1 M2 --tag glue_v0
  python experiments/lineage_glue.py eval --tag glue_v0 --eval M3 M4
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE  # noqa: E402

INDEX = ROOT / "outputs/index/one_in_a_million"
CKPT = RESULTS / "glue"
D_IN = 19
# cuDNN SDPA re-autotunes for every new sequence length (~10x slower here).
torch.backends.cuda.enable_cudnn_sdp(False)


# ----------------------------------------------------------------------------- data
class Movie:
    """Per-frame cell tables and lineage for one OIAM movie (training only)."""

    def __init__(self, alias: str):
        seq = SEQUENCE[alias]
        obs = pd.read_parquet(INDEX / seq / "observations.parquet",
                              columns=["frame_index", "gt_track_id", "center_y", "center_x"])
        shape = pd.read_parquet(INDEX / seq / "shape.parquet")
        obs = obs.merge(shape, on=["frame_index", "gt_track_id"], validate="one_to_one")
        tracks = pd.read_parquet(INDEX / seq / "tracks.parquet")
        frames = pd.read_parquet(INDEX / seq / "frames.parquet")
        self.image_shape = tuple(int(v) for v in str(frames["shape"].iloc[0]).strip("()").split(","))
        self.start = dict(zip(tracks.gt_track_id, tracks.start_frame))
        self.parent = dict(zip(tracks.gt_track_id, tracks.parent_id))
        self.frames = {}
        for t, g in obs.groupby("frame_index"):
            self.frames[int(t)] = {
                "track": g.gt_track_id.to_numpy(),
                "pos": g[["center_y", "center_x"]].to_numpy(np.float64),
                "axis": g[["axis_y", "axis_x"]].to_numpy(np.float64),
                "major": g.major_sd.to_numpy(np.float64),
                "minor": g.minor_sd.to_numpy(np.float64)}
        self.valid_a = [t for t, f in self.frames.items() if len(f["track"]) >= 2]

    def ancestor(self, track: int, a: int) -> int:
        while self.start.get(track, -1) > a:
            track = self.parent.get(track, 0)
            if track == 0:
                return 0
        return track

    def window(self, a: int, b: int):
        fa, fb = self.frames[a], self.frames[b]
        col = {int(t): i for i, t in enumerate(fa["track"])}
        label = np.array([col.get(self.ancestor(int(t), a), -1) for t in fb["track"]])
        return fa, fb, label


def rotation(rng):
    th = rng.uniform(0, 2 * math.pi)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    if rng.random() < 0.5:
        r = r @ np.diag([1.0, -1.0])
    return r


def border_vector(pos, image_shape):
    h, w = image_shape
    cand = np.stack([np.stack([-pos[:, 0], 0 * pos[:, 1]], 1),
                     np.stack([h - pos[:, 0], 0 * pos[:, 1]], 1),
                     np.stack([0 * pos[:, 0], -pos[:, 1]], 1),
                     np.stack([0 * pos[:, 0], w - pos[:, 1]], 1)], 1)
    k = np.abs(cand).sum(-1).argmin(1)
    return cand[np.arange(len(pos)), k]


def spacing_of(pos):
    if len(pos) < 2:
        return 10.0
    return max(float(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1])), 1.0)


def featurize(fa, fb, horizon, image_shape, rot=None, moved=None):
    """Return anchor/target token features and canonical positions (in target spacings).

    `moved` (optional, pixels) replaces the radial-expansion initial guess of
    each anchor's descendant centre, e.g. a Darcy-flow advection."""
    rot = np.eye(2) if rot is None else rot
    pa, pb = fa["pos"], fb["pos"]
    na, nb = len(pa), len(pb)
    s = spacing_of(pb)
    growth = nb / max(na, 1)
    centre = np.array(image_shape, dtype=np.float64) / 2
    mu_a, mu_b = pa.mean(0), pb.mean(0)
    if moved is None:
        can_a = (pa - mu_a) * math.sqrt(growth) / s @ rot.T
    else:
        can_a = (moved - mu_b) / s @ rot.T
    can_b = (pb - mu_b) / s @ rot.T
    glob = np.array([math.log2(horizon / 60), math.log(growth), math.log(na), math.log(nb)])
    shift = (mu_b - mu_a) / s @ rot.T

    def tokens(f, pos, can, n):
        axis = f["axis"] / np.maximum(np.linalg.norm(f["axis"], axis=1, keepdims=True), 1e-6)
        axis = axis @ rot.T
        ang2 = np.stack([axis[:, 0] ** 2 - axis[:, 1] ** 2, 2 * axis[:, 0] * axis[:, 1]], 1)
        bv = border_vector(pos, image_shape) / s @ rot.T
        bd = np.linalg.norm(bv, axis=1, keepdims=True)
        absolute = (pos - centre) / s @ rot.T
        return np.column_stack([
            can, absolute / 10, ang2, np.log(np.maximum(f["major"], 1e-3))[:, None] - 2,
            np.log(np.maximum(f["minor"], 1e-3))[:, None] - 1.4,
            np.tanh(bv / 4), np.log1p(bd), np.repeat(shift[None], n, 0) / 10,
            np.repeat(glob[None], n, 0), np.ones((n, 1)), np.zeros((n, 1))])

    xa = tokens(fa, pa, can_a, na)
    xb = tokens(fb, pb, can_b, nb)
    xb[:, -2:] = [0, 1]
    return (torch.tensor(xa, dtype=torch.float32), torch.tensor(xb, dtype=torch.float32),
            torch.tensor(can_a, dtype=torch.float32), torch.tensor(can_b, dtype=torch.float32))


# ----------------------------------------------------------------------------- model
class Fourier(nn.Module):
    def __init__(self, d, scales=(0.02, 0.1, 0.5, 2.0)):
        super().__init__()
        g = torch.randn(2, d // (2 * len(scales)), generator=torch.Generator().manual_seed(0))
        self.register_buffer("b", torch.cat([g * s for s in scales], 1))

    def forward(self, p):
        z = 2 * math.pi * p @ self.b
        return torch.cat([z.sin(), z.cos()], -1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.sa_a = nn.MultiheadAttention(d, heads, batch_first=True)
        self.sa_b = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca_a = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca_b = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(6)])
        self.ff_a = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ff_b = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, a, b):
        n = self.norms
        x = n[0](a); a = a + self.sa_a(x, x, x, need_weights=False)[0]
        y = n[1](b); b = b + self.sa_b(y, y, y, need_weights=False)[0]
        x, y = n[2](a), n[3](b)
        a = a + self.ca_a(x, y, y, need_weights=False)[0]
        b = b + self.ca_b(y, x, x, need_weights=False)[0]
        a = a + self.ff_a(n[4](a))
        b = b + self.ff_b(n[5](b))
        return a, b


class LineageGlue(nn.Module):
    def __init__(self, d=192, heads=6, layers=6):
        super().__init__()
        self.inp = nn.Linear(D_IN, d)
        self.pe = Fourier(d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.out_a = nn.LayerNorm(d)
        self.out_b = nn.LayerNorm(d)
        self.centre = nn.Linear(d, 3)   # offset (2) + log sigma
        self.key = nn.Linear(d, d)
        self.query = nn.Linear(d, d)
        self.dust = nn.Linear(d, 1)
        self.family = nn.Linear(d, 64)  # target embedding for same-family head
        nn.init.zeros_(self.centre.weight)
        nn.init.zeros_(self.centre.bias)

    def forward(self, xa, xb, can_a, can_b):
        a = self.inp(xa) + self.pe(can_a)
        b = self.inp(xb) + self.pe(can_b)
        a, b = a[None], b[None]
        for blk in self.blocks:
            a, b = blk(a, b)
        a, b = self.out_a(a)[0], self.out_b(b)[0]
        c = self.centre(a)
        centre = can_a + c[:, :2]
        log_sigma = c[:, 2].clamp(-2, 3)
        d2 = ((can_b[:, None] - centre[None]) ** 2).sum(-1)
        geo = -0.5 * d2 / torch.exp(2 * log_sigma)[None] - 2 * log_sigma[None]
        sim = self.query(b) @ self.key(a).T / math.sqrt(a.shape[-1])
        logits = torch.cat([geo + sim, self.dust(b)], 1)
        return logits, centre, F.normalize(self.family(b), dim=-1)


# ----------------------------------------------------------------------------- train
def sample(movies, rng, h_range=(8, 240)):
    for _ in range(100):  # a movie shorter than the sampled horizons is skipped for another one
        m = movies[rng.integers(len(movies))]
        for _ in range(100):
            h = int(round(math.exp(rng.uniform(math.log(h_range[0]), math.log(h_range[1])))))
            a = int(rng.choice(m.valid_a))
            if a + h in m.frames:
                fa, fb, label = m.window(a, a + h)
                return m, fa, fb, label, h
    raise RuntimeError("no window")


def family_loss(emb, label, rng, n_pairs=4096):
    ok = np.flatnonzero(label >= 0)
    if len(ok) < 2:
        return emb.sum() * 0
    i = torch.as_tensor(rng.choice(ok, n_pairs), device=emb.device)
    j = torch.as_tensor(rng.choice(ok, n_pairs), device=emb.device)
    lab = torch.as_tensor(label, device=emb.device)
    y = (lab[i] == lab[j]).float()
    return F.binary_cross_entropy_with_logits(10 * (emb[i] * emb[j]).sum(-1) - 2, y)


def train(args):
    dev = "cuda"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    movies = [Movie(a) for a in args.train]
    print("loaded", args.train, flush=True)
    model = LineageGlue(args.d, args.heads, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    CKPT.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    for step in range(args.steps):
        m, fa, fb, label, h = sample(movies, rng)
        xa, xb, ca, cb = (t.to(dev) for t in featurize(fa, fb, h, m.image_shape, rotation(rng)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, centre, emb = model(xa, xb, ca, cb)
        logits = logits.float()
        y = torch.as_tensor(np.where(label >= 0, label, logits.shape[1] - 1), device=dev)
        loss = F.cross_entropy(logits, y)
        floss = family_loss(emb.float(), label, rng) if args.family else loss * 0
        total = loss + args.family * floss
        opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == y)[torch.as_tensor(label >= 0, device=dev)].float().mean()
        log.append((loss.item(), floss.item(), acc.item(), h, len(label)))
        if (step + 1) % 500 == 0:
            arr = np.array(log[-500:])
            print(f"step {step+1} ce={arr[:,0].mean():.3f} fam={arr[:,1].mean():.3f} "
                  f"acc={arr[:,2].mean():.3f} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % 5000 == 0 or step + 1 == args.steps:
            torch.save({"model": model.state_dict(), "args": vars(args)}, CKPT / f"{args.tag}.pt")


# ----------------------------------------------------------------------------- eval
def load_model(tag, dev="cuda"):
    ck = torch.load(CKPT / f"{tag}.pt", map_location=dev, weights_only=False)
    a = ck["args"]
    model = LineageGlue(a["d"], a["heads"], a["layers"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def frame_dict(df):
    return {"pos": df[["center_y", "center_x"]].to_numpy(np.float64),
            "axis": df[["axis_y", "axis_x"]].fillna(0).to_numpy(np.float64),
            "major": df.major_sd.fillna(1).to_numpy(np.float64),
            "minor": df.minor_sd.fillna(1).to_numpy(np.float64)}


@torch.no_grad()
def predict(model, e, image_shape, tta=1, dev="cuda"):
    fa, fb = frame_dict(e["anchor"]), frame_dict(e["target"])
    h = e["window"]["horizon_frames"]
    rng = np.random.default_rng(0)
    prob = 0
    for k in range(tta):
        rot = None if k == 0 else rotation(rng)
        xa, xb, ca, cb = (t.to(dev) for t in featurize(fa, fb, h, image_shape, rot))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, centre, emb = model(xa, xb, ca, cb)
        prob = prob + torch.softmax(logits.float(), 1)
    prob = (prob / tta).cpu().numpy()
    return prob, emb.float().cpu().numpy()


def image_shape_of(alias):
    frames = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet")
    return tuple(int(v) for v in str(frames["shape"].iloc[0]).strip("()").split(","))


def evaluate(args):
    from cellanc.loader import Windows
    from experiments.family_set import examples
    from experiments.snapshot_lineage_ceiling import cluster_match  # noqa: F401
    model = load_model(args.tag)
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in args.eval:
        shape = image_shape_of(alias)
        for h in args.horizons:
            for e in examples(alias, windows, compensate=False, horizon=h):
                prob, emb = predict(model, e, shape, args.tta)
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok):
                    continue
                r = ok.target_detection_id.map(ro).to_numpy()
                truth = ok.ancestor_detection_id.map(co).to_numpy()
                pos = e["target"][["center_y", "center_x"]].to_numpy(float)
                for name, pred in decoders(prob, emb, pos).items():
                    pred = pred[r]
                    rows.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"],
                                 "n_anchors": len(co), "n_scored": len(r),
                                 "method": f"{args.tag}_{name}",
                                 "accuracy": float(np.mean(pred == truth)),
                                 "set_f1": set_f1(truth, pred)})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"glue_eval_{args.tag}.parquet", index=False)
    report(df)


def knn_graph(pos, k=12):
    n = len(pos)
    if n < 2:
        return np.zeros((0,), int), np.zeros((0,), int)
    nn_ = np.asarray(cKDTree(pos).query(pos, k=min(k + 1, n))[1]).reshape(n, -1)[:, 1:]
    return np.repeat(np.arange(n), nn_.shape[1]), nn_.ravel()


def mean_field(prob, i, j, w, beta, iters=10):
    """q_i  propto  p_i * exp(beta * sum_k W_ik q_k), W row-normalised on the kNN graph."""
    logp = np.log(np.maximum(prob, 1e-9))
    deg = np.bincount(i, w, minlength=len(prob)) + 1e-9
    q = prob
    for _ in range(iters):
        msg = np.zeros_like(q)
        np.add.at(msg, i, w[:, None] * q[j])
        z = logp + beta * msg / deg[:, None]
        z -= z.max(1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(1, keepdims=True)
    return q


def cluster_assign(prob, i, j, w, tau):
    """Union target pairs with same-family prob > tau, then one anchor per cluster."""
    parent = np.arange(len(prob))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k in np.argsort(-w):
        if w[k] < tau:
            break
        a, b = find(i[k]), find(j[k])
        if a != b:
            parent[a] = b
    root = np.array([find(x) for x in range(len(prob))])
    logp = np.log(np.maximum(prob, 1e-9))
    out = np.empty(len(prob), int)
    for c in np.unique(root):
        m = root == c
        out[m] = logp[m].sum(0).argmax()
    return out


def decoders(prob, emb, pos):
    p = prob[:, :-1]
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
    out = {"argmax": p.argmax(1)}
    i, j = knn_graph(pos)
    names = [f"mf_{w}{b}" for b in (1, 3) for w in ("fam", "self")] + \
        [f"cl_fam{t}" for t in (0.5, 0.7, 0.9)]
    if len(i) == 0:
        return {**out, **{n: out["argmax"] for n in names}}
    w_fam = 1 / (1 + np.exp(-(10 * (emb[i] * emb[j]).sum(1) - 2)))
    w_self = (p[i] * p[j]).sum(1)
    for beta in (1, 3):
        out[f"mf_fam{beta}"] = mean_field(p, i, j, w_fam, beta).argmax(1)
        out[f"mf_self{beta}"] = mean_field(p, i, j, w_self, beta).argmax(1)
    for tau in (0.5, 0.7, 0.9):
        out[f"cl_fam{tau}"] = cluster_assign(p, i, j, w_fam, tau)
    return out


def set_f1(truth, pred):
    anchors = np.union1d(truth, pred)
    f = [2 * np.sum((truth == k) & (pred == k)) / (np.sum(truth == k) + np.sum(pred == k))
         for k in anchors]
    return float(np.mean(f))


def report(df):
    df = df.assign(dense=df.n_anchors > 200)
    print(df.groupby(["method", "H"])[["accuracy", "set_f1"]].mean().round(4).to_string())
    print("dense", df[df.dense].groupby(["method", "H"])[["accuracy", "set_f1"]].mean().round(4).to_string())
    print(df.groupby(["method", "H", "sequence"]).accuracy.mean().unstack().round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--train", nargs="+", default=["M0", "M1", "M2"])
    p.add_argument("--eval", nargs="+", default=["M3", "M4"])
    p.add_argument("--horizons", nargs="+", type=int, default=[33, 66, 132])
    p.add_argument("--tag", default="glue_v0")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--family", type=float, default=0.0)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    (train if args.mode == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
