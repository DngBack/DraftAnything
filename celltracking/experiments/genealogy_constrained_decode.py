"""Genealogy-constrained family registration (docs/genealogy_constrained_protocol.md).

Joint inference of the ancestor assignment pi and the descendant-family partition it induces, from the same network
scores as the paper decoder (three-seed LineageRAFT ensemble raft_v3*, snapshot-genealogy affinities):

  E(pi) = - sum_j log p_{j,pi(j)}  -  lam * sum_{(j,k) in E} [s_jk log w_jk + (1 - s_jk) log(1 - w_jk)]
          + mu * sum_i phi_i(n_i),       s_jk = 1[pi(j) = pi(k)],  n_i = |{j : pi(j) = i}|

phi_i is a binary-fission growth prior on the visible family size: a discretised Gaussian with mean m_i = R v_i and
variance kappa m_i, where R is the window's per-ancestor growth factor, estimated without labels from the network's
soft family sizes of interior ancestors, and v_i is the fraction of ancestor i's family that lies inside the image,
computed from its family centre c_i. The family centre is coupled to the assignment: after each decode the centre of
every non-empty family is re-estimated from its members and the network re-scores candidates relative to it.

Inference: mean field on the pairwise term with the cardinality term linearised at the current expected sizes, then
exact MAP of (pairwise field + unaries + convex cardinality) by min-cost bipartite matching over per-ancestor slots.

  python experiments/genealogy_constrained_decode.py collect    # GPU: network scores + affinities + labels, cached
  python experiments/genealogy_constrained_decode.py decode     # decoders on the cache (+ GPU re-scoring)
  python experiments/genealogy_constrained_decode.py summary
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile
import torch
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.family import family_edges  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.lineage_glue import INDEX, Movie, featurize, frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import initial_centres, knn, load  # noqa: E402
from experiments.partition_metrics import genealogy, pair_f1  # noqa: E402
from experiments.raft_family_probe import match_groups  # noqa: E402
from experiments.snapshot_genealogy import THRESHOLDS, patches  # noqa: E402

SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
HS = (33, 66, 132)
CACHE = RESULTS / "genealogy_constrained_cache.pkl"
EPS = 1e-4


# ----------------------------------------------------------------------------- network scores
@torch.no_grad()
def raft_scores(model, xa, xb, ca, cb, centre_override=None):
    """LineageRAFT forward; optionally replace the final family centres before candidate scoring.

    Returns dense target x anchor probabilities (dustbin dropped, not renormalised), dustbin probabilities and the
    final centres (canonical units)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ha, hb = model.inp_a(xa), model.inp_b(xb)
        ia = knn(ca, ca, model.k_self + 1)
        ib = knn(cb, cb, model.k_self + 1)
        for la, lb in zip(model.enc_a, model.enc_b):
            ha = la(ha, ha, ia, ca[ia] - ca[:, None])
            hb = lb(hb, hb, ib, cb[ib] - cb[:, None])
        ha, hb = model.ctx_a(ha), model.ctx_b(hb)
        centre = ca
        for t in range(model.iters):
            c = centre.detach()
            nb = knn(c, cb, model.k_cross)
            ha = model.a_from_b[t](ha, hb, nb, cb[nb] - c[:, None])
            ha = model.a_from_a[t](ha, ha, ia, ca[ia] - ca[:, None])
            pred = c + model.delta[t](ha)
            centre = pred if model.move else c
            na = knn(cb, centre.detach(), model.k_cross)
            hb = model.b_from_a[t](hb, ha, na, centre.detach()[na] - cb[:, None])
        centre = centre.float()
        if centre_override is not None:
            centre = centre_override
        cand = knn(cb, centre, model.k_out)
        off = cb[:, None] - centre[cand]
        pair_in = torch.cat([hb[:, None].expand(-1, cand.shape[1], -1), ha[cand], off / 4,
                             off.norm(dim=-1, keepdim=True) / 4], -1)
        logits = torch.cat([model.pair(pair_in).squeeze(-1), model.dust(hb)], 1)
    p = torch.softmax(logits.float(), 1)
    dense = torch.zeros(len(xb), len(xa), device=xa.device)
    dense.scatter_add_(1, cand, p[:, :-1])
    return dense.cpu().numpy(), p[:, -1].cpu().numpy(), centre.cpu().numpy()


def featurized(e, h, shape, dev="cuda"):
    fa, fb = frame_dict(e["anchor"]), frame_dict(e["target"])
    moved = initial_centres(fa, fb, h, shape, "darcy")
    return tuple(t.to(dev) for t in featurize(fa, fb, h, shape, None, moved))


@torch.no_grad()
def affinities_all(net, mt, st, img, df, dev, k=8):
    """Genealogy probabilities P(tau < t) for every threshold t on the k-NN target graph (flip-averaged)."""
    e, x = family_edges(df, k)
    pos = df[["center_y", "center_x"]].to_numpy(np.float32)
    out = []
    for s in range(0, len(e), 512):
        ek = e[s:s + 512]
        xg = (torch.as_tensor(x[s:s + 512], device=dev) - mt) / st
        z = 0.5 * (net(xg, patches(img, pos, ek, dev)).float() +
                   net(xg, patches(img, pos, ek[:, ::-1].copy(), dev)).float())
        out.append(torch.sigmoid(z).cpu().numpy())
    return e[:, 0], e[:, 1], np.concatenate(out) if out else np.zeros((0, len(THRESHOLDS)), np.float32)


def collect():
    dev = "cuda"
    models = [load(t)[0] for t in SEEDS]
    net, mt, st = genealogy("snapshot_genealogy.pt", dev)
    windows = Windows(BENCH, ROOT / "data").windows
    out = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        movie = Movie(alias)
        paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
        for h in HS:
            for e in examples(alias, windows, compensate=False, horizon=h):
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): k for k, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): k for k, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok) or len(co) < 2 or len(e["target"]) < 9:
                    continue
                xa, xb, ca, cb = featurized(e, h, shape, dev)
                t0 = time.time()
                seeds = [raft_scores(m, xa, xb, ca, cb) for m in models]
                torch.cuda.synchronize()
                t_net = time.time() - t0
                im = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
                im = (im - im.mean()) / (im.std() + 1e-6)
                df = e["target"].reset_index(drop=True)
                gi, gj, gw = affinities_all(net, mt, st, im, df, dev)
                # evaluator-only labels: true ancestor of every target and its ancestor H/2 earlier (sub-family)
                a, b = int(e["window"]["anchor_frame"]), int(e["window"]["target_frame"])
                col = {int(t): i for i, t in enumerate(e["anchor"].gt_track_id)}
                tracks = e["target"].gt_track_id.to_numpy()
                lab_all = np.array([col.get(movie.ancestor(int(t), a), -1) for t in tracks])
                sub = np.array([movie.ancestor(int(t), b - h // 2) for t in tracks])
                out.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"], "n_anchors": len(co),
                            "shape": shape, "pa": e["anchor"][["center_y", "center_x"]].to_numpy(np.float32),
                            "pb": df[["center_y", "center_x"]].to_numpy(np.float32),
                            "ca": ca.cpu().numpy(), "cb": cb.cpu().numpy(),
                            "p": [sp.csr_matrix(s[0]) for s in seeds], "dust": [s[1] for s in seeds],
                            "centre": [s[2] for s in seeds], "t_net": t_net,
                            "gi": gi, "gj": gj, "gw": gw.astype(np.float32),
                            "r": ok.target_detection_id.map(ro).to_numpy(),
                            "truth": ok.ancestor_detection_id.map(co).to_numpy(),
                            "lab_all": lab_all, "sub": sub})
            print(alias, h, len(out), flush=True)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


# ----------------------------------------------------------------------------- growth prior
def pixel_frame(w):
    """Canonical -> pixel map of the target frame: pb = cb * s + mu (fitted, exact up to float error)."""
    s = float(np.median(np.linalg.norm(w["pb"] - w["pb"].mean(0), axis=1) /
                        np.maximum(np.linalg.norm(w["cb"] - w["cb"].mean(0), axis=1), 1e-6)))
    mu = (w["pb"] - w["cb"] * s).mean(0)
    return s, mu


def visibility(centre_px, shape, sigma):
    """Fraction of an isotropic Gaussian family blob (std sigma, px) centred at each centre inside the image."""
    from scipy.special import ndtr
    v = np.ones(len(centre_px))
    for ax, size in enumerate(shape[:2]):
        c = centre_px[:, ax]
        v *= ndtr((size - c) / sigma) - ndtr(-c / sigma)
    return np.clip(v, 1e-3, 1.0)


def growth_prior(w, centre_can, n_dust, fano, use_vis=True, poisson=False):
    """Per-ancestor mean m_i and variance s2_i of the visible family size (no labels).

    R from mass conservation: sum_i R v_i = N_b - expected entering cells. Visible size = binomial thinning of the
    full family (Fano factor `fano` from RodSim): var = v^2 fano R + R v (1 - v)."""
    s, mu = pixel_frame(w)
    n_b = len(w["pb"])
    r0 = n_b / len(w["pa"])
    if use_vis:
        sigma = 0.5 * s * np.sqrt(max(r0, 1.0))
        v = visibility(centre_can * s + mu, w["shape"], sigma)
    else:
        v = np.ones(len(centre_can))
    R = max(n_b - n_dust, 1.0) / v.sum()
    m = R * v
    s2 = m if poisson else v ** 2 * fano * R + R * v * (1 - v)
    return m, np.maximum(s2, 0.25), R


def fano_of(R, cfg):
    return max(cfg["fano_intercept"] + cfg["fano_slope_log2R"] * np.log2(max(R, 1.0)), 0.05)


# ----------------------------------------------------------------------------- inference
def dense_p(w, seeds=(0, 1, 2), override=None):
    ps = override if override is not None else [w["p"][k].toarray() for k in seeds]
    p = np.mean(ps, 0)
    return p / np.maximum(p.sum(1, keepdims=True), 1e-9)


def pair_field(q, i, j, theta):
    m = np.zeros_like(q)
    np.add.at(m, i, theta[:, None] * q[j])
    np.add.at(m, j, theta[:, None] * q[i])
    return m


def size_cost(n, m, s2):
    """phi(n+1) - phi(n) for phi(n) = (n - m)^2 / (2 s2): marginal cost of one more member."""
    return (2 * (n - m) + 1) / (2 * s2)


def mean_field_gc(logp, i, j, theta, lam, mu, m, s2, iters=10, damp=0.5):
    """Mean field on lam * pairwise + mu * linearised cardinality; returns q and the pairwise field."""
    q = np.exp(logp - logp.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    for _ in range(iters):
        f = lam * pair_field(q, i, j, theta) if lam else 0.0
        z = logp + f
        if mu:
            nbar = q.sum(0)[None] - q  # expected size of each family without target j
            z = z - mu * size_cost(nbar, m[None], s2[None])
        z = np.exp(z - z.max(1, keepdims=True))
        z /= z.sum(1, keepdims=True)
        q = damp * q + (1 - damp) * z
    f = lam * pair_field(q, i, j, theta) if lam else np.zeros_like(q)
    return q, f


def map_cardinality(u, mu, m, s2, topk=12, extra=4.0):
    """Exact argmax_pi sum_j u[j, pi(j)] - mu * sum_i phi_i(n_i) for convex phi_i, by min-cost bipartite matching
    of targets to per-ancestor slots (slot s of ancestor i costs mu * (phi_i(s) - phi_i(s-1)), non-decreasing).
    Targets are restricted to their top-k ancestors by u; a private fallback slot (cost +1e3) keeps it feasible."""
    nb, na = u.shape
    if mu == 0:
        return u.argmax(1), 0
    k = min(topk, na)
    cand = np.argpartition(-u, k - 1, axis=1)[:, :k]
    slots = np.ceil(m + extra * np.sqrt(s2) + 2).astype(int)
    slots = np.minimum(slots, nb)
    off = np.concatenate([[0], np.cumsum(slots)])
    rows, cols, vals = [], [], []
    for s in range(int(slots.max())):
        ok = slots[cand] > s
        jj, kk = np.nonzero(ok)
        ii = cand[jj, kk]
        rows.append(jj)
        cols.append(off[ii] + s)
        vals.append(-u[jj, ii] + mu * size_cost(s, m[ii], s2[ii]))
    jj = np.arange(nb)
    rows.append(jj)
    cols.append(off[-1] + jj)
    vals.append(-u.max(1) + 1e3)
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    vals = vals - vals.min() + 1.0  # strictly positive: explicit entries are edges
    g = sp.csr_matrix((vals, (rows, cols)), shape=(nb, off[-1] + nb))
    row, col = min_weight_full_bipartite_matching(g)
    match = np.empty(nb, int)
    match[row] = col
    fallback = match >= off[-1]
    anc = np.searchsorted(off, match, side="right") - 1
    anc[fallback] = u[fallback].argmax(1)
    return anc, int(fallback.sum())


def family_centres(cb, pi, n_a, prior):
    """Decoded family centroids (canonical); empty families keep the network's centre."""
    c = prior.copy()
    n = np.bincount(pi, minlength=n_a)
    for ax in range(2):
        s = np.bincount(pi, weights=cb[:, ax], minlength=n_a)
        c[n > 0, ax] = s[n > 0] / n[n > 0]
    return c


# ----------------------------------------------------------------------------- metrics
def metrics(pred, w):
    r, t = w["r"], w["truth"]
    pr = pred[r]
    n_a = len(w["pa"])
    nt = np.bincount(t, minlength=n_a)
    npd = np.bincount(pr, minlength=n_a)
    act = nt > 0
    nz = npd > 0
    frag = np.mean([len(np.unique(pr[t == a])) for a in np.flatnonzero(act)])
    return {"acc": float(np.mean(pr == t)), "f1": float(pair_f1(t, pr)[2]),
            "size_err": float(np.abs(npd - nt)[act].mean()), "size_std": float(npd[nz].std()),
            "singleton": float(np.mean(npd[nz] == 1)), "frag": float(frag)}


# ----------------------------------------------------------------------------- decisive comparison
def structured(w, logp, th, lam, mu, centre, dust, cfg, use_vis=True, poisson=False):
    n_a = logp.shape[1]
    m, s2, _ = growth_prior(w, centre, dust, fano_of(len(w["pb"]) / n_a, cfg), use_vis, poisson)
    q, f = mean_field_gc(logp, w["gi"], w["gj"], th, lam, mu, m, s2)
    pi, n_fb = map_cardinality(logp + f, mu, m, s2)
    return pi, n_fb


def decode():
    import json
    from experiments.image_partition_decode import mean_field
    cfg = json.loads((RESULTS / "growth_prior_sim.json").read_text())
    with open(CACHE, "rb") as f:
        items = pickle.load(f)
    models = [load(t)[0] for t in SEEDS]
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        for h in HS:
            ex = {e["window"]["window_id"]: e for e in examples(alias, windows, compensate=False, horizon=h)}
            for w in (x for x in items if x["sequence"] == alias and x["H"] == h):
                n_a = len(w["pa"])
                p = dense_p(w)
                logp = np.log(np.maximum(p, 1e-9))
                wv = np.clip(w["gw"][:, THRESHOLDS.index(h)], EPS, 1 - EPS)
                th = np.log(wv / (1 - wv))
                centre = np.mean(w["centre"], 0)
                dust = float(np.mean([d.sum() for d in w["dust"]]))
                la = w["lab_all"]
                n_true = np.bincount(la[la >= 0], minlength=n_a).astype(float)
                fams = np.unique(la[la >= 0])
                f_true = np.stack([w["cb"][la == a].mean(0) for a in fams])
                preds, times, fb = {}, {}, {}

                def timed(name, fn):
                    t0 = time.time()
                    out = fn()
                    times[name] = time.time() - t0
                    preds[name], fb[name] = out if isinstance(out, tuple) else (out, 0)

                timed("argmax", lambda: p.argmax(1))
                timed("potts", lambda: mean_field(logp, w["gi"], w["gj"], wv, 2.0))
                timed("growth_only", lambda: structured(w, logp, th, 0.0, 1.0, centre, dust, cfg))
                timed("pairwise_only", lambda: structured(w, logp, th, 1.0, 0.0, centre, dust, cfg))
                timed("gc_noEM", lambda: structured(w, logp, th, 1.0, 1.0, centre, dust, cfg))
                timed("gc_novis", lambda: structured(w, logp, th, 1.0, 1.0, centre, dust, cfg, use_vis=False))
                timed("gc_poisson", lambda: structured(w, logp, th, 1.0, 1.0, centre, dust, cfg, poisson=True))

                def potts_growth():
                    m, s2, _ = growth_prior(w, centre, dust, fano_of(len(w["pb"]) / n_a, cfg))
                    q, f = mean_field_gc(logp, w["gi"], w["gj"], 2 * wv - 1, 2.0, 1.0, m, s2)
                    return map_cardinality(logp + f, 1.0, m, s2)

                timed("potts_growth", potts_growth)

                def full():
                    e = ex[w["window_id"]]
                    xa, xb, ca, cb = featurized(e, h, shape)
                    cen = [family_centres(w["cb"], preds["gc_noEM"], n_a, c) for c in w["centre"]]
                    res = [raft_scores(mm, xa, xb, ca, cb, torch.as_tensor(c, device="cuda", dtype=torch.float32))
                           for mm, c in zip(models, cen)]
                    p2 = dense_p(w, override=[r[0] for r in res])
                    lp2 = np.log(np.maximum(p2, 1e-9))
                    return structured(w, lp2, th, 1.0, 1.0, np.mean([r[2] for r in res], 0),
                                      float(np.mean([r[1].sum() for r in res])), cfg)

                timed("gc_full", full)
                times["gc_full"] += times["gc_noEM"]
                timed("oracle_card", lambda: map_cardinality(
                    logp + mean_field_gc(logp, w["gi"], w["gj"], th, 1.0, 0.0, n_true, np.full(n_a, .25))[1],
                    5.0, n_true, np.full(n_a, 0.25)))
                timed("oracle_partition", lambda: match_groups(logp, np.where(la >= 0, la, n_a + np.arange(len(la)))))
                cerr = float(np.median(np.linalg.norm(centre[fams] - f_true, axis=1)))
                for name, pr in preds.items():
                    rows.append({"sequence": alias, "H": h, "window_id": w["window_id"], "n_anchors": w["n_anchors"],
                                 "decoder": name, "seconds": times[name], "fallback": fb[name],
                                 "t_net": w["t_net"], "centre_err": cerr, **metrics(pr, w)})
                tr = metrics(np.where(la >= 0, la, 0), w)
                rows.append({"sequence": alias, "H": h, "window_id": w["window_id"], "n_anchors": w["n_anchors"],
                             "decoder": "truth", "seconds": 0.0, "fallback": 0, "t_net": w["t_net"],
                             "centre_err": cerr, **tr})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "genealogy_constrained_decode.parquet", index=False)
    return df


def boot(a, b, n=5000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    s = d[idx].mean(1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def summary(df=None):
    if df is None:
        df = pd.read_parquet(RESULTS / "genealogy_constrained_decode.parquet")
    df["dense"] = df.n_anchors > 200
    cols = ["acc", "f1", "size_err", "size_std", "singleton", "frag", "seconds"]
    for subset, part in (("pooled", df), ("M4 (confirmatory)", df[df.sequence == "M4"]),
                         ("M3 (development)", df[df.sequence == "M3"])):
        for dense in (False, True):
            s = part[part.dense] if dense else part
            print(f"\n== {subset}, {'dense' if dense else 'all'} windows ==")
            print(s.groupby(["H", "decoder"])[cols].mean().round(3).unstack("H").to_string())
            for h in HS:
                ref = s[(s.H == h) & (s.decoder == "potts")].set_index("window_id")
                for name in ("gc_noEM", "gc_full"):
                    new = s[(s.H == h) & (s.decoder == name)].set_index("window_id").loc[ref.index]
                    if len(ref) < 3:
                        continue
                    print(f"H={h} {name} - potts (n={len(ref)}): " + "; ".join(
                        "%s %+.3f [%+.3f, %+.3f]" % ((m,) + boot(new[m], ref[m])) for m in
                        ("acc", "f1", "size_err", "singleton")))
    print("fallback slots used:", int(df.fallback.sum()))


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
    elif sys.argv[1:] == ["decode"]:
        summary(decode())
    elif sys.argv[1:] == ["summary"]:
        summary()
