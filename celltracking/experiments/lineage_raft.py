"""LineageRAFT: iterative family-centre registration + local matching.

Each anchor carries a predicted descendant-family centre c_j, initialised by a
parameter-free radial expansion. For T iterations: anchors attend to the
targets nearest to c_j (relative offsets), anchors exchange messages with
anchor neighbours (flow smoothing), c_j is updated, and targets attend to the
nearest centres. Target->anchor logits are an MLP over each target's K nearest
centres (+ dustbin). Deep supervision on c_j uses GT family centroids
(training only). A same-family head scores target kNN pairs.

  python experiments/lineage_raft.py train --tag raft_v1
  python experiments/lineage_raft.py eval --tag raft_v1
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import BENCH, RESULTS, ROOT  # noqa: E402
from experiments.lineage_glue import (D_IN, Movie, featurize, frame_dict, image_shape_of,  # noqa: E402
                                      knn_graph, mean_field, rotation, sample, set_f1)

CKPT = RESULTS / "glue"


def knn(q, x, k):
    k = min(k, len(x))
    return torch.cdist(q, x).topk(k, largest=False).indices


class LocalAttn(nn.Module):
    """Query i attends to gathered neighbours with relative-offset encoding."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.rel = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.o = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, hq, hx, idx, rel):
        n, k = idx.shape
        d = hq.shape[-1]
        q = self.q(self.n1(hq)).view(n, self.h, d // self.h)
        kv = self.kv(self.n2(hx))[idx] + self.rel(rel / 4)
        key, val = kv.view(n, k, 2, self.h, d // self.h).unbind(2)
        att = torch.einsum("nhc,nkhc->nhk", q, key) / math.sqrt(d // self.h)
        out = torch.einsum("nhk,nkhc->nhc", att.softmax(-1), val).reshape(n, d)
        hq = hq + self.o(out)
        return hq + self.ff(hq)


class GlobalCtx(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mix = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, h):
        return h + self.mix(torch.cat([h, h.mean(0, keepdim=True).expand_as(h)], -1))


class LineageRAFT(nn.Module):
    def __init__(self, d=128, iters=4, k_self=12, k_cross=24, k_out=24, move=True, aux=False, couple=False):
        super().__init__()
        # couple: each round also runs an E-step (ancestors compete for the targets near their centres) and feeds the
        # M-step (claimed-family centroid and soft size) to the centre update, so centre and partition are inferred
        # jointly (docs/coupled_refinement_protocol.md). Zero-initialised: at step 0 identical to the plain model.
        self.couple = couple
        self.iters, self.k_self, self.k_cross, self.k_out = iters, k_self, k_cross, k_out
        self.move = move  # ablation: False keeps every family centre at its initialisation
        self.aux = aux  # ablation: with move=False, still predict (and supervise) centres but never use them
        self.inp_a = nn.Sequential(nn.Linear(D_IN, d), nn.GELU(), nn.Linear(d, d))
        self.inp_b = nn.Sequential(nn.Linear(D_IN, d), nn.GELU(), nn.Linear(d, d))
        self.enc_a = nn.ModuleList([LocalAttn(d) for _ in range(2)])
        self.enc_b = nn.ModuleList([LocalAttn(d) for _ in range(2)])
        self.ctx_a, self.ctx_b = GlobalCtx(d), GlobalCtx(d)
        self.a_from_b = nn.ModuleList([LocalAttn(d) for _ in range(iters)])
        self.a_from_a = nn.ModuleList([LocalAttn(d) for _ in range(iters)])
        self.b_from_a = nn.ModuleList([LocalAttn(d) for _ in range(iters)])
        self.delta = nn.ModuleList([nn.Linear(d, 2) for _ in range(iters)])
        for lin in self.delta:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.pair = nn.Sequential(nn.Linear(2 * d + 3, d), nn.GELU(), nn.Linear(d, d),
                                  nn.GELU(), nn.Linear(d, 1))
        self.dust = nn.Linear(d, 1)
        self.fam = nn.Sequential(nn.Linear(2 * d + 3, d), nn.GELU(), nn.Linear(d, 1))
        if couple:
            self.claim = nn.ModuleList([nn.Sequential(nn.Linear(2 * d + 3, d), nn.GELU(), nn.Linear(d, 1))
                                        for _ in range(iters)])
            self.m_in = nn.ModuleList([nn.Linear(4, d) for _ in range(iters)])
            for lin in self.m_in:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def em_step(self, t, ha, hb, nb, c, can_b):
        """E: responsibility of anchor i for target nb[i, k], normalised over all anchors claiming that target.
        M: soft family size n_i and claimed centroid m_i, returned as a residual input to the anchor state."""
        off = can_b[nb] - c[:, None]
        x = torch.cat([ha[:, None].expand(-1, nb.shape[1], -1), hb[nb], off / 4, off.norm(dim=-1, keepdim=True) / 4], -1)
        s = self.claim[t](x).squeeze(-1).float()
        flat, sf = nb.reshape(-1), s.reshape(-1)
        mx = torch.full((len(can_b),), -1e9, device=s.device).scatter_reduce(0, flat, sf, "amax")
        e = torch.exp(sf - mx[flat])
        z = torch.zeros(len(can_b), device=s.device).index_add_(0, flat, e)
        q = (e / z[flat].clamp_min(1e-9)).view_as(s)
        n = q.sum(1, keepdim=True)
        m = (q[..., None] * can_b[nb].float()).sum(1) / n.clamp_min(1e-6)
        g = n / (n + 1)
        return ha + self.m_in[t](torch.cat([g * (m - c.float()) / 4, torch.log1p(n), g], -1).to(ha.dtype))

    def forward(self, xa, xb, can_a, can_b):
        ha, hb = self.inp_a(xa), self.inp_b(xb)
        ia = knn(can_a, can_a, self.k_self + 1)
        ib = knn(can_b, can_b, self.k_self + 1)
        for la, lb in zip(self.enc_a, self.enc_b):
            ha = la(ha, ha, ia, can_a[ia] - can_a[:, None])
            hb = lb(hb, hb, ib, can_b[ib] - can_b[:, None])
        ha, hb = self.ctx_a(ha), self.ctx_b(hb)
        centre = can_a
        centres = []
        for t in range(self.iters):
            c = centre.detach()
            nb = knn(c, can_b, self.k_cross)
            ha = self.a_from_b[t](ha, hb, nb, can_b[nb] - c[:, None])
            ha = self.a_from_a[t](ha, ha, ia, can_a[ia] - can_a[:, None])
            if self.couple:
                ha = self.em_step(t, ha, hb, nb, c, can_b)
            pred = c + self.delta[t](ha)
            centre = pred if self.move else c
            if self.move or self.aux:
                centres.append(pred)
            na = knn(can_b, centre.detach(), self.k_cross)
            hb = self.b_from_a[t](hb, ha, na, centre.detach()[na] - can_b[:, None])
        cand = knn(can_b, centre.detach(), self.k_out)
        off = can_b[:, None] - centre[cand]
        pair_in = torch.cat([hb[:, None].expand(-1, cand.shape[1], -1), ha[cand], off / 4,
                             off.norm(dim=-1, keepdim=True) / 4], -1)
        logits = torch.cat([self.pair(pair_in).squeeze(-1), self.dust(hb)], 1)
        return logits, cand, centres, hb

    def family_logits(self, hb, can_b, i, j):
        off = can_b[j] - can_b[i]
        x1 = torch.cat([hb[i], hb[j], off / 4, off.norm(dim=-1, keepdim=True) / 4], -1)
        x2 = torch.cat([hb[j], hb[i], -off / 4, off.norm(dim=-1, keepdim=True) / 4], -1)
        return 0.5 * (self.fam(x1) + self.fam(x2)).squeeze(-1)


def targets_from(label, cand, n_cand):
    """Index of the true anchor among candidates; dustbin if no ancestor; -100 if missed."""
    lab = torch.as_tensor(label, device=cand.device)
    hit = cand == lab[:, None]
    y = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(lab, -100))
    return torch.where(lab < 0, torch.full_like(lab, n_cand), y)


def initial_centres(fa, fb, h, image_shape, init):
    if init != "darcy" or len(fa["pos"]) < 2:
        return None
    from experiments.darcy_flow_prior import advect
    return advect(fa["pos"], fb["pos"], h, image_shape)


class WindowStream(torch.utils.data.IterableDataset):
    def __init__(self, movies, args):
        self.movies, self.args = movies, args

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.args.seed, wid])
        a = self.args
        while True:
            m, fa, fb, label, h = sample(self.movies, rng, (a.hmin, a.hmax))
            rot = rotation(rng) if a.rot == "full" else (
                np.diag([1.0, rng.choice([-1.0, 1.0])]) if a.rot == "flip" else None)
            moved = initial_centres(fa, fb, h, m.image_shape, a.init)
            xa, xb, ca, cb = featurize(fa, fb, h, m.image_shape, rot, moved)
            if a.no_shape:
                xa[:, 4:8] = 0
                xb[:, 4:8] = 0
            yield xa, xb, ca, cb, torch.as_tensor(label), h


def load_movies(args):
    if args.pseudo:  # annotation-free: lineage from chained one-frame links
        from experiments.chained_pseudo_lineage import PseudoMovie
        movies = [PseudoMovie(a) for a in args.train]
    else:
        movies = [Movie(a) for a in args.train]
    if args.sim:
        from experiments.colony_sim import SimMovie
        paths = sorted(Path(args.sim).glob("colony_*.pkl"))[:args.n_sim]
        movies += [SimMovie(q, args.sim_full_fov) for q in paths]
        print("sim colonies", len(paths), flush=True)
    return movies


def train(args):
    dev = "cuda"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    movies = load_movies(args)
    stream = iter(torch.utils.data.DataLoader(WindowStream(movies, args), batch_size=None,
                                              num_workers=args.workers, prefetch_factor=8))
    model = LineageRAFT(args.d, args.iters, k_out=args.k_out, move=not args.no_centre, aux=args.aux_centre,
                        couple=args.couple).to(dev)
    if args.init_from:
        model.load_state_dict(torch.load(CKPT / f"{args.init_from}.pt", map_location=dev)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    CKPT.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    for step in range(args.steps):
        xa, xb, ca, cb, label, h = next(stream)
        xa, xb, ca, cb = (t.to(dev, non_blocking=True) for t in (xa, xb, ca, cb))
        label = label.numpy()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, cand, centres, hb = model(xa, xb, ca, cb)
        logits = logits.float()
        y = targets_from(label, cand, cand.shape[1])
        ce = F.cross_entropy(logits, y, ignore_index=-100)
        # GT family centroids in canonical target coordinates
        lab = torch.as_tensor(label, device=dev)
        ok = lab >= 0
        cnt = torch.zeros(len(ca), device=dev).index_add_(0, lab[ok], torch.ones(int(ok.sum()), device=dev))
        tot = torch.zeros(len(ca), 2, device=dev).index_add_(0, lab[ok], cb[ok])
        has = cnt > 0
        gt_c = tot[has] / cnt[has, None]
        closs = sum((0.8 ** (len(centres) - 1 - t) * (c.float()[has] - gt_c).norm(dim=-1).clamp(max=10).mean()
                     for t, c in enumerate(centres)), torch.zeros((), device=dev))
        fl = ce * 0
        if args.family and ok.sum() > 2:
            i, j = knn_graph(cb.cpu().numpy(), 8)
            i, j = torch.as_tensor(i, device=dev), torch.as_tensor(j, device=dev)
            keep = ok[i] & ok[j]
            i, j = i[keep], j[keep]
            if len(i):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fz = model.family_logits(hb, cb, i, j)
                fl = F.binary_cross_entropy_with_logits(fz.float(), (lab[i] == lab[j]).float())
                if args.consist:
                    # partition consistency of the assignment itself: sum_j p_ij p_kj = 1[same]
                    pr = torch.softmax(logits, 1)
                    dense = torch.zeros(len(cb), len(ca) + 1, device=dev)
                    dense.scatter_add_(1, cand, pr[:, :-1])
                    dense[:, -1] = pr[:, -1]
                    overlap = (dense[i] * dense[j]).sum(1).clamp(1e-5, 1 - 1e-5)
                    cl = F.binary_cross_entropy(overlap, (lab[i] == lab[j]).float())
                    fl = fl + args.consist / max(args.family, 1e-6) * cl
        loss = ce + args.closs * closs + args.family * fl
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            pred = logits.argmax(1)
            good = torch.as_tensor(label >= 0, device=dev)
            yy = torch.where(y < 0, torch.full_like(y, -1), y)
            acc = (pred == yy)[good].float().mean()
            recall = (y != -100)[good].float().mean()
        log.append((ce.item(), float(closs.detach()), fl.item(), acc.item(), recall.item()))
        if (step + 1) % 500 == 0:
            a = np.array(log[-500:])
            print(f"step {step+1} ce={a[:,0].mean():.3f} centre={a[:,1].mean():.3f} fam={a[:,2].mean():.3f} "
                  f"acc={a[:,3].mean():.3f} recall={a[:,4].mean():.3f} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % 5000 == 0 or step + 1 == args.steps:
            torch.save({"model": model.state_dict(), "args": vars(args)}, CKPT / f"{args.tag}.pt")


def load(tag, dev="cuda"):
    ck = torch.load(CKPT / f"{tag}.pt", map_location=dev, weights_only=False)
    a = ck["args"]
    model = LineageRAFT(a["d"], a["iters"], k_out=a.get("k_out", 24), move=not a.get("no_centre", False),
                        aux=a.get("aux_centre", False), couple=a.get("couple", False)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, a


@torch.no_grad()
def predict(model, fa, fb, h, image_shape, no_shape=False, tta=1, dev="cuda", init="radial"):
    """Dense target x (anchor + dustbin) probabilities and target-family edge probs."""
    rng = np.random.default_rng(0)
    prob = 0
    moved = initial_centres(fa, fb, h, image_shape, init)
    for k in range(tta):
        rot = None if k == 0 else rotation(rng)
        xa, xb, ca, cb = (t.to(dev) for t in featurize(fa, fb, h, image_shape, rot, moved))
        if no_shape:
            xa[:, 4:8] = 0
            xb[:, 4:8] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, cand, centres, hb = model(xa, xb, ca, cb)
        p = torch.softmax(logits.float(), 1)
        dense = torch.zeros(len(xb), len(xa) + 1, device=dev)
        dense.scatter_add_(1, cand, p[:, :-1])
        dense[:, -1] = p[:, -1]
        prob = prob + dense
        if k == 0:
            i, j = knn_graph(cb.cpu().numpy(), 12)
            ti, tj = torch.as_tensor(i, device=dev), torch.as_tensor(j, device=dev)
            if len(ti):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fam = torch.sigmoid(model.family_logits(hb, cb, ti, tj).float()).cpu().numpy()
            else:
                fam = np.zeros(0)
            centre = (centres[-1] if centres else ca).float().cpu().numpy()
    return (prob / tta).cpu().numpy(), (i, j, fam), centre


def decoders(prob, fam):
    p = prob[:, :-1]
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
    out = {"argmax": p.argmax(1)}
    i, j, w = fam
    for beta in (1, 2, 4):
        if len(i):
            out[f"mf_fam{beta}"] = mean_field(p, i, j, w, beta).argmax(1)
            out[f"mf_self{beta}"] = mean_field(p, i, j, (p[i] * p[j]).sum(1), beta).argmax(1)
        else:
            out[f"mf_fam{beta}"] = out[f"mf_self{beta}"] = out["argmax"]
    return out


def evaluate(args):
    from cellanc.loader import Windows
    from experiments.family_set import examples
    model, margs = load(args.tag)
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in args.eval:
        shape = image_shape_of(alias)
        for h in args.horizons:
            for e in examples(alias, windows, compensate=False, horizon=h):
                prob, fam, _ = predict(model, frame_dict(e["anchor"]), frame_dict(e["target"]),
                                       h, shape, margs.get("no_shape", False), args.tta,
                                       init=margs.get("init", "radial"))
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok):
                    continue
                r = ok.target_detection_id.map(ro).to_numpy()
                truth = ok.ancestor_detection_id.map(co).to_numpy()
                for name, pred in decoders(prob, fam).items():
                    pred = pred[r]
                    rows.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"],
                                 "n_anchors": len(co), "n_scored": len(r),
                                 "method": f"{args.tag}_{name}",
                                 "accuracy": float(np.mean(pred == truth)),
                                 "set_f1": set_f1(truth, pred)})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"raft_eval_{args.tag}.parquet", index=False)
    df = df.assign(dense=df.n_anchors > 200)
    print(df.groupby(["method", "H"]).accuracy.mean().unstack().round(4).to_string())
    print("dense\n", df[df.dense].groupby(["method", "H"]).accuracy.mean().unstack().round(4).to_string())
    print(df.groupby(["method", "H"]).set_f1.mean().unstack().round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--train", nargs="*", default=["M0", "M1", "M2"])
    p.add_argument("--sim", default="")
    p.add_argument("--n-sim", type=int, default=100000)
    p.add_argument("--eval", nargs="+", default=["M3", "M4"])
    p.add_argument("--horizons", nargs="+", type=int, default=[33, 66, 132])
    p.add_argument("--tag", default="raft_v1")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--closs", type=float, default=0.3)
    p.add_argument("--family", type=float, default=0.5)
    p.add_argument("--consist", type=float, default=0.0)
    p.add_argument("--hmin", type=int, default=8)
    p.add_argument("--hmax", type=int, default=240)
    p.add_argument("--no-shape", action="store_true")
    p.add_argument("--rot", choices=["full", "flip", "none"], default="full")
    p.add_argument("--init", choices=["radial", "darcy"], default="radial")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-full-fov", action="store_true", help="ablation: no FOV crop in simulation")
    p.add_argument("--init-from", default="", help="checkpoint tag to fine-tune from")
    p.add_argument("--pseudo", action="store_true", help="train on chained pseudo-lineage, no annotation")
    p.add_argument("--no-centre", action="store_true", help="ablation: family centres stay at their initialisation")
    p.add_argument("--aux-centre", action="store_true", help="ablation: with --no-centre, keep the centre head and loss unused")
    p.add_argument("--k-out", type=int, default=24, help="candidate ancestors per target")
    p.add_argument("--couple", action="store_true", help="E/M coupling of partition and centre in each round")
    args = p.parse_args()
    (train if args.mode == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
