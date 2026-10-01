"""Direction A pilot: coarse-grain genealogies by the ancestry they imply (OIAM, H=132).

A history G is one parent choice per cell per kept-frame link; the query is q(G) = ancestor at
frame a of every cell at b. Instead of storing histories, the lumped state keeps per cell
  A[anchor, cell] = P(ancestor | O)      and (item 4 of the proposal) a growth state
  pi[cell, k]     = P(k links since the cell's lineage last divided; k = link index if never).
The age state sets the expected supply of each parent in the next link (mu[link][k], fitted on
train GT), so histories with the same ancestry but different division timing stay apart.

Same link model everywhere (semi-relaxed OT of cellanc.transport, eps/tau from the earlier tune):
  direct_ends       endpoints only, no intermediate frames (direct ancestry scoring)
  lumped            product of per-link P(parent | child); no growth state, uniform mass
  sampled_S         S histories drawn from the same links, ancestry counted; -> lumped as S grows
  lumped_age        lumped + mean-field age state pi
  sampled_age_S     S histories, each carries its exact age state, one OT solve per history/link
  oracle_age        lumped with GT age classes: bound on what the age state can buy in this model
  lumped_area       lumped with mask-area mass (oracle masks): observed proxy for the age state
  pm_lap_T_S        perturb-and-MAP on the capacity-4 LAP (coupled assignment sampling, the
                    PyUAT-style control): Gumbel noise of scale T, S solves per link
PyUAT's own hard-assignment accuracy is in outputs/results/pyuat_h132_M3.parquet. Trackastra is
not installed here and is not run.

  PYTHONPATH=. python experiments/coarse_grain_pilot.py [--seqs M3] [--schedules s32 s16 u5]
    -> outputs/results/coarse_grain_h132_{targets,windows}.parquet, _config.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"  # one process per window instead

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import expansion_compensate  # noqa: E402
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.lineage import trace_to_anchor  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from cellanc.transport import log_plan  # noqa: E402
from main import BENCH, DATA, OUT  # noqa: E402

SEQ = {f"M{i}": f"0{i}" for i in range(5)}
EPS, TAU = 0.1, 3.0  # outputs/results/decide_h132_tune.parquet
FLOOR = 1e-3  # posterior mixed with uniform over anchors before NLL, same for every method
S_PLAIN, S_AGE, S_LAP, T_LAP = (1, 4, 16, 64, 256), (1, 4, 16), (4, 16), (0.3, 1.0)
_state = {}
_K: dict = {}  # per-process kernel cache, one entry per link of the current window
_WS = Windows(BENCH, DATA)


def _load(alias):
    if alias not in _state:
        idx = OUT / "index/one_in_a_million" / SEQ[alias]
        tr = pd.read_parquet(idx / "tracks.parquet")
        obs = pd.read_parquet(idx / "observations.parquet",
                              columns=["frame_index", "gt_track_id", "marker_pixels"])
        gm = pd.read_parquet(BENCH / "labels/gt_mapping" / f"{alias}.parquet").merge(
            obs, on=["frame_index", "gt_track_id"])
        tg = pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
        _state[alias] = {
            "tracks": dict(zip(tr.gt_track_id, zip(tr.start_frame, tr.end_frame, tr.parent_id))),
            "present": obs.groupby("frame_index").gt_track_id.apply(set).to_dict(),
            "local": {(t, i): (g, a) for t, i, g, a in zip(
                gm.frame_index, gm.local_detection_id, gm.gt_track_id, gm.marker_pixels)},
            "targets": dict(list(tg.groupby("window_id")))}
    return _state[alias]


def gt_class(u, i, frames, tracks):
    """Links since track u's lineage last divided, seen at frames[i]; i if not since frames[0]."""
    s, _, p = tracks[u]
    return sum(f >= s for f in frames[:i]) if p else i


def fit_mu(ws):
    """mu[link][k] = mean number of descendants at the next kept frame, per age class (train GT)."""
    acc = defaultdict(list)
    for w in ws:
        st, fr = _load(w["sequence_id"]), w["observed_frames"]
        for i, (f0, f1) in enumerate(zip(fr, fr[1:])):
            cnt = defaultdict(int)
            for u in st["present"][f1]:
                cnt[trace_to_anchor(u, f0, st["tracks"], st["present"][f0])[0]] += 1
            for u in st["present"][f0]:
                acc[(i, gt_class(u, i, fr, st["tracks"]))].append(cnt[u])
    n_links = max(i for i, _ in acc) + 1
    mu = []
    for i in range(n_links):
        link = np.concatenate([acc[(i, k)] for k in range(i + 1)]).mean()
        # ponytail: classes with < 20 train cells fall back to the link mean
        mu.append([float(np.mean(acc[(i, k)])) if len(acc[(i, k)]) >= 20 else float(link)
                   for k in range(i + 1)])
    return mu


def cond(src, dst, m_src, m_dst=None, iters=1000, tol=1e-4):
    """P(parent | child), (n_src, n_dst), columns sum to 1. Same fixed point as
    transport.log_plan, iterated on exp(-cost/eps) with matrix products instead of logsumexp
    (about 100x faster on 1500-cell frames; selfcheck compares the two)."""
    if len(src) < 2:
        return np.full((len(src), len(dst)), 1.0 / len(src))
    key = (src.shape, dst.shape, src[0, 1], dst[0, 1])
    if key not in _K:
        d = expansion_compensate(src, dst)[None, :, 1:] - src[:, None, 1:]
        spacing2 = np.median(cKDTree(src[:, 1:]).query(src[:, 1:], k=2)[0][:, 1]) ** 2
        _K[key] = np.maximum(np.exp(-(d ** 2).sum(-1) / (spacing2 * EPS)), 1e-300)
    K = _K[key]
    b = np.ones(len(dst)) if m_dst is None else m_dst
    a = np.maximum(m_src, 1e-6)
    a = a * (b.sum() / a.sum())
    f, lu = TAU / (TAU + EPS), np.zeros(len(src))
    for _ in range(iters):
        lu_new = f * np.log(a / (K @ (b / (np.exp(lu) @ K))))
        done = np.abs(lu_new - lu).max() < tol
        lu = lu_new
        if done:
            break
    P = np.exp(lu)[:, None] * K
    return P / P.sum(0, keepdims=True)


def sample_parents(cum, rng):
    """One parent row index per column, from cum = C.cumsum(0)."""
    return np.minimum((cum < rng.random(cum.shape[1])).sum(0), len(cum) - 1)


def age_step(C, pi):
    """Mean-field update of the age state through one link. A child is 'born' (class 0) when its
    parent has another child; otherwise it inherits the parent's class + 1."""
    Cc = np.minimum(C, 1 - 1e-9)
    L = np.log1p(-Cc)
    alone = np.exp(L.sum(1, keepdims=True) - L)  # P(no other child of i | j is a child of i)
    out = np.zeros((C.shape[1], pi.shape[1] + 1))
    out[:, 1:] = (C * alone).T @ pi
    out[:, 0] = 1 - out[:, 1:].sum(1)
    return out


def counts_to_post(anc, n_a):
    """(S, n_b) sampled anchor row indices -> (n_a, n_b) empirical posterior."""
    P = np.zeros((n_a, anc.shape[1]))
    np.add.at(P, (anc, np.arange(anc.shape[1])[None, :]), 1.0 / len(anc))
    return P


def lap_cost(src, dst, cap=4):
    """baselines._lap cost in spacing units, (n_dst, cap * n_src); None where _lap falls back."""
    d2 = ((dst[:, None, 1:] - src[None, :, 1:]) ** 2).sum(-1)
    if len(src) < 2 or len(dst) > cap * len(src):
        return None, d2.argmin(1)
    d2 = d2 / np.median(cKDTree(src[:, 1:]).query(src[:, 1:], k=2)[0][:, 1]) ** 2
    return np.concatenate([d2 + k for k in range(cap)], axis=1), None


def lap_sample(cost, nearest, n_src, T, rng):
    """Parent row index per dst row: MAP of the LAP cost minus T * Gumbel noise."""
    if cost is None:
        return nearest
    if T:
        cost = cost - T * rng.gumbel(size=cost.shape)
    rows, cols = linear_sum_assignment(cost)
    parent = np.empty(len(cost), dtype=int)
    parent[rows] = cols % n_src
    return parent


def run(job):
    w, mu, parts = job
    part = Path(parts) / f"{w['window_id']}.pkl"
    if part.exists():  # resume: finished windows are kept across restarts
        return pd.read_pickle(part)
    st = _load(w["sequence_id"])
    win = _WS.load(w)
    det, fr = win["detections"], w["observed_frames"]
    links = list(zip(fr, fr[1:]))
    _K.clear()
    n_a, n_b = len(det[fr[0]]), len(det[fr[-1]])
    ones = {t: np.ones(len(d)) for t, d in det.items()}
    area = {t: np.array([st["local"][(t, int(i))][1] for i in d[:, 0]], float) for t, d in det.items()}
    rng = np.random.default_rng(0)
    post, cost = {}, {}  # method -> (n_a, n_b) posterior; method -> (seconds, stored state entries)

    t0 = time.perf_counter()
    post["direct_ends"] = cond(det[fr[0]], det[fr[-1]], ones[fr[0]])
    cost["direct_ends"] = (time.perf_counter() - t0, n_a * n_b)

    t0 = time.perf_counter()
    Cs = [cond(det[p], det[c], ones[p]) for p, c in links]
    t_plans = time.perf_counter() - t0
    t0 = time.perf_counter()
    A = np.eye(n_a)
    for C in Cs:
        A = A @ C
    post["lumped"] = A
    cost["lumped"] = (t_plans + time.perf_counter() - t0, n_a * max(len(det[t]) for t in fr))

    t0 = time.perf_counter()
    anc = np.tile(np.arange(n_a), (max(S_PLAIN), 1))
    for C in Cs:
        cum = C.cumsum(0)
        anc = np.stack([a[sample_parents(cum, rng)] for a in anc])
    t_s = time.perf_counter() - t0
    for S in S_PLAIN:  # nested prefixes of the same draws; time is linear in S
        post[f"sampled_{S}"] = counts_to_post(anc[:S], n_a)
        cost[f"sampled_{S}"] = (t_plans + t_s * S / max(S_PLAIN), S * max(len(det[t]) for t in fr))

    t0 = time.perf_counter()
    A, pi = np.eye(n_a), np.ones((n_a, 1))
    for i, (p, c) in enumerate(links):
        C = cond(det[p], det[c], pi @ np.array(mu[i]))
        A, pi = A @ C, age_step(C, pi)
    post["lumped_age"] = A
    cost["lumped_age"] = (time.perf_counter() - t0, (n_a + len(fr)) * max(len(det[t]) for t in fr))

    t0 = time.perf_counter()
    anc = []
    for _ in range(max(S_AGE)):
        a, k = np.arange(n_a), np.zeros(n_a, dtype=int)
        for i, (p, c) in enumerate(links):
            par = sample_parents(cond(det[p], det[c], np.array(mu[i])[k]).cumsum(0), rng)
            born = np.bincount(par, minlength=len(k))[par] >= 2
            a, k = a[par], np.where(born, 0, k[par] + 1)
        anc.append(a)
    t_s, anc = time.perf_counter() - t0, np.stack(anc)
    for S in S_AGE:
        post[f"sampled_age_{S}"] = counts_to_post(anc[:S], n_a)
        cost[f"sampled_age_{S}"] = (t_s * S / max(S_AGE), 2 * S * max(len(det[t]) for t in fr))

    t0 = time.perf_counter()
    A = np.eye(n_a)
    for i, (p, c) in enumerate(links):
        k = [gt_class(st["local"][(p, int(j))][0], i, fr, st["tracks"]) for j in det[p][:, 0]]
        A = A @ cond(det[p], det[c], np.array(mu[i])[k])
    post["oracle_age"] = A
    cost["oracle_age"] = (time.perf_counter() - t0, n_a * max(len(det[t]) for t in fr))

    t0 = time.perf_counter()
    A = np.eye(n_a)
    for p, c in links:
        A = A @ cond(det[p], det[c], area[p], area[c])
    post["lumped_area"] = A
    cost["lumped_area"] = (time.perf_counter() - t0, n_a * max(len(det[t]) for t in fr))

    t0 = time.perf_counter()
    lc = [lap_cost(det[p], det[c]) for p, c in links]
    t_cost = time.perf_counter() - t0
    for T in (0.0, *T_LAP):
        t0 = time.perf_counter()
        n_s = max(S_LAP) if T else 1
        anc = np.tile(np.arange(n_a), (n_s, 1))
        for (p, _), (cm, nn) in zip(links, lc):
            anc = np.stack([a[lap_sample(cm, nn, len(det[p]), T, rng)] for a in anc])
        t_s = t_cost + time.perf_counter() - t0
        for S in (S_LAP if T else (1,)):
            name = f"pm_lap_T{T}_{S}" if T else "map_lap_c4"
            post[name] = counts_to_post(anc[:S], n_a)
            cost[name] = (t_s * S / n_s, S * max(len(det[t]) for t in fr))

    tg = st["targets"][w["window_id"]]
    ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
    a_ids, b_ids = det[fr[0]][:, 0].astype(int), det[fr[-1]][:, 0].astype(int)
    col = pd.Series(np.arange(n_b), b_ids)[ok.target_detection_id.to_numpy()].to_numpy()
    row = pd.Series(np.arange(n_a), a_ids)[ok.ancestor_detection_id.to_numpy().astype(int)].to_numpy()
    sib = pd.DataFrame(columns=["det_1", "det_2"])
    base = {"window_id": w["window_id"], "sequence_id": w["sequence_id"],
            "schedule_id": w["schedule_id"], "n_anchors": n_a}
    t_rows, w_rows = [], []
    for name, P in post.items():
        P = (1 - FLOOR) * P + FLOOR / n_a
        t_rows.append(pd.DataFrame({**base, "method": name, "conf": P[:, col].max(0),
                                    "correct": P[:, col].argmax(0) == row,
                                    "p_true": P[row, col]}))
        s = score_window(dict(zip(b_ids, a_ids[P.argmax(0)])), set(), tg, sib)
        w_rows.append({**base, "method": name, "seconds": cost[name][0], "state": cost[name][1],
                       **{k: s[k] for k in ["n_scored", "n_correct", "set_f1_sum", "count_n_anchors"]}})
    pd.to_pickle((pd.concat(t_rows), w_rows), part)
    print("done", w["window_id"], n_a, n_b, flush=True)
    return pd.concat(t_rows), w_rows


def selfcheck():
    C = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # parent 0 divides, parent 1 does not
    pi = age_step(C, np.ones((2, 1)))
    assert np.allclose(pi, [[1, 0], [1, 0], [0, 1]], atol=1e-6), pi
    rng = np.random.default_rng(0)
    C = rng.dirichlet(np.ones(4), size=5).T
    D = rng.dirichlet(np.ones(5), size=6).T
    anc = np.stack([np.arange(4)[sample_parents(C.cumsum(0), rng)][sample_parents(D.cumsum(0), rng)] for _ in range(20000)])
    assert np.abs(counts_to_post(anc, 4) - C @ D).max() < 0.02  # sampling estimates the lumped product
    src = np.column_stack([np.arange(30), rng.random((30, 2)) * 100])
    dst = np.column_stack([np.arange(45), rng.random((45, 2)) * 100])
    m = rng.random(30) + 0.5
    lp = log_plan(src, dst, m, np.ones(45), EPS, TAU)
    assert np.abs(cond(src, dst, m) - np.exp(lp - logsumexp(lp, axis=0, keepdims=True))).max() < 1e-3
    _K.clear()


def summarize(Tg, W):
    W = W.assign(acc=W.n_correct / W.n_scored.replace(0, np.nan),
                 f1=W.set_f1_sum / W.count_n_anchors.replace(0, np.nan))
    out = W.groupby(["schedule_id", "method"]).agg(
        acc=("acc", "mean"), set_f1=("f1", "mean"), seconds=("seconds", "sum"), state=("state", "max"))

    def post_quality(g):
        g = g.sort_values("conf", ascending=False)
        bins = np.minimum((g.conf * 10).astype(int), 9)
        ece = sum(abs(b.correct.mean() - b.conf.mean()) * len(b) for _, b in g.groupby(bins)) / len(g)
        return pd.Series({"pooled_acc": g.correct.mean(), "nll": -np.log(g.p_true).mean(), "ece": ece,
                          "acc_cov50": g.correct.iloc[:len(g) // 2].mean()})
    return out.join(Tg.groupby(["schedule_id", "method"])[["conf", "correct", "p_true"]].apply(post_quality))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["M3"])
    ap.add_argument("--train", nargs="+", default=["M0", "M1", "M2"])
    ap.add_argument("--H", type=int, default=132)
    ap.add_argument("--schedules", nargs="+", default=["s32", "s16", "u5"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    selfcheck()

    def pick(seqs, sid):
        return [w for w in _WS.windows if w["sequence_id"] in seqs and w["horizon_frames"] == args.H
                and w["schedule_id"] == sid and f"A_h{args.H}" in w["experiments"]]
    mu = {sid: fit_mu(pick(args.train, sid)) for sid in args.schedules}
    stem = OUT / "results" / f"coarse_grain_h{args.H}{args.tag}"
    Path(f"{stem}_parts").mkdir(exist_ok=True)
    jobs = [(w, mu[sid], f"{stem}_parts") for sid in args.schedules for w in pick(args.seqs, sid)]
    jobs.sort(key=lambda j: -j[0]["anchor_frame"])  # biggest windows first
    jobs = jobs[:args.limit]
    print(len(jobs), "windows", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, jobs))  # ponytail: mu is not part of the resume key; clear _parts after changing --train
    Tg = pd.concat(r[0] for r in res)
    W = pd.DataFrame([x for r in res for x in r[1]])
    Tg.to_parquet(f"{stem}_targets.parquet", index=False)
    W.to_parquet(f"{stem}_windows.parquet", index=False)
    Path(f"{stem}_config.json").write_text(json.dumps(
        {"eps": EPS, "tau": TAU, "floor": FLOOR, "train": args.train, "eval": args.seqs, "mu": mu}, indent=1))
    pd.set_option("display.width", 250)
    print(summarize(Tg, W).round(3).to_string())
