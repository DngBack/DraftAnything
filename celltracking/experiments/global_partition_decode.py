"""Non-local family partitioners on OIAM M3/M4 (docs/global_partition_protocol.md).

Does a stronger, non-local partitioner than the 8-NN Potts decoder close the gap to the oracle family partition?
Correlation clustering (GAEC + greedy node moves) on image-affinity log-odds (P1) or image + LineageRAFT co-assignment
log-odds (P2), and spectral clustering with K = number of distinct argmax ancestors (P3), on 8- and 24-NN target
graphs, with predicted and (reference) true affinities. Each partition is assigned to ancestors by cluster + argmax
(free) and Hungarian (hung). Base: three-seed ensemble raft_v3, raft_v3_s1, raft_v3_s2.

  python experiments/global_partition_decode.py            # GPU collect + cache + CPU decode + summary
  python experiments/global_partition_decode.py decode     # decode + summary from cache
"""

import heapq
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile
import torch
from sklearn.cluster import SpectralClustering

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.family import family_edges  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.image_partition_decode import free_match, mean_field  # noqa: E402
from experiments.lineage_glue import INDEX, frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.partition_metrics import genealogy, pair_f1  # noqa: E402
from experiments.raft_family_probe import match_groups  # noqa: E402
from experiments.snapshot_genealogy import THRESHOLDS, patches  # noqa: E402

SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
HS = (66, 132)
KS = (8, 24)
CACHE = RESULTS / "global_partition_decode_cache.pkl"
EPS = 1e-4


@torch.no_grad()
def affinity_k(net, mt, st, img, df, h, dev, k):
    """image_affinity on the k-NN target graph (edge features do not depend on k)."""
    e, x = family_edges(df, k)
    pos = df[["center_y", "center_x"]].to_numpy(np.float32)
    t = THRESHOLDS.index(h)
    out = []
    for s in range(0, len(e), 512):
        ek = e[s:s + 512]
        xg = (torch.as_tensor(x[s:s + 512], device=dev) - mt) / st
        z = 0.5 * (net(xg, patches(img, pos, ek, dev)).float() +
                   net(xg, patches(img, pos, ek[:, ::-1].copy(), dev)).float())
        out.append(torch.sigmoid(z[:, t]).cpu().numpy())
    return e[:, 0], e[:, 1], np.concatenate(out) if out else np.zeros(0)


def collect():
    dev = "cuda"
    models = [load(t) for t in SEEDS]
    net, mt, st = genealogy("snapshot_genealogy.pt", dev)
    windows = Windows(BENCH, ROOT / "data").windows
    out = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
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
                prob = np.mean([predict(m, frame_dict(e["anchor"]), frame_dict(e["target"]), h, shape,
                                        ma.get("no_shape", False), init=ma.get("init", "radial"))[0]
                                for m, ma in models], 0)
                p = prob[:, :-1] / np.maximum(prob[:, :-1].sum(1, keepdims=True), 1e-9)
                im = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
                im = (im - im.mean()) / (im.std() + 1e-6)
                df = e["target"].reset_index(drop=True)
                out.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"], "n_anchors": len(co),
                            "p": p.astype(np.float32), "r": ok.target_detection_id.map(ro).to_numpy(),
                            "truth": ok.ancestor_detection_id.map(co).to_numpy(),
                            "graphs": {k: affinity_k(net, mt, st, im, df, h, dev, k) for k in KS}})
            print(alias, h, flush=True)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def gaec(n, i, j, theta):
    """Greedy additive edge contraction for correlation clustering (maximise summed intra-cluster theta)."""
    adj = [dict() for _ in range(n)]
    for x, y, v in zip(i, j, theta):
        if x != y:
            adj[x][y] = adj[x].get(y, 0.0) + v
            adj[y][x] = adj[y].get(x, 0.0) + v
    ver = np.zeros(n, int)
    alive = np.ones(n, bool)
    heap = [(-v, x, y, 0, 0) for x in range(n) for y, v in adj[x].items() if x < y and v > 0]
    heapq.heapify(heap)
    members = {x: [x] for x in range(n)}
    while heap:
        _, x, y, vx, vy = heapq.heappop(heap)
        if not (alive[x] and alive[y]) or ver[x] != vx or ver[y] != vy:
            continue
        if len(adj[x]) < len(adj[y]):
            x, y = y, x
        for c, v in adj[y].items():
            if c != x:
                adj[x][c] = adj[x].get(c, 0.0) + v
                adj[c][x] = adj[x][c]
            adj[c].pop(y, None)
        adj[x].pop(y, None)
        adj[y] = {}
        alive[y] = False
        members[x] += members.pop(y)
        ver[x] += 1
        for c, v in adj[x].items():
            if v > 0:
                heapq.heappush(heap, (-v, min(x, c), max(x, c), ver[min(x, c)], ver[max(x, c)]))
    lab = np.empty(n, int)
    for g, (_, m) in enumerate(members.items()):
        lab[m] = g
    return lab


def node_moves(n, i, j, theta, lab, passes=50):
    """KL-style greedy local search: move a node to the adjacent cluster (or a new singleton) of largest gain."""
    nb = [[] for _ in range(n)]
    for x, y, v in zip(i, j, theta):
        if x != y:
            nb[x].append((y, v))
            nb[y].append((x, v))
    lab = lab.copy()
    size = np.bincount(lab, minlength=n + lab.max() + 1).astype(int)
    size = np.concatenate([size, np.zeros(n, int)])
    nxt = lab.max() + 1
    for _ in range(passes):
        moved = False
        for v in range(n):
            s = {}
            for c, w in nb[v]:
                s[lab[c]] = s.get(lab[c], 0.0) + w
            own = s.pop(lab[v], 0.0)
            best, gain = None, 1e-9
            for d, val in s.items():
                if val - own > gain:
                    best, gain = d, val - own
            if best is None and size[lab[v]] > 1 and -own > gain:
                best, gain = nxt, -own
                nxt += 1
            if best is not None:
                size[lab[v]] -= 1
                size[best] += 1
                lab[v] = best
                moved = True
        if not moved:
            break
    return np.unique(lab, return_inverse=True)[1]


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def spectral(n, i, j, w, k):
    a = sp.coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    k = int(min(max(k, 2), n - 1))
    return SpectralClustering(k, affinity="precomputed", assign_labels="cluster_qr", random_state=0).fit_predict(a)


def decode_window(it):
    p, r, truth = it["p"].astype(np.float64), it["r"], it["truth"]
    n = len(p)
    logp = np.log(np.maximum(p, 1e-9))
    labels = np.full(n, -1)
    labels[r] = truth
    k_spec = len(np.unique(p.argmax(1)))
    preds, parts, fails = {"argmax": p.argmax(1)}, {}, 0
    for k in KS:
        i, j, wp = it["graphs"][k]
        known = (labels[i] >= 0) & (labels[j] >= 0)
        wt = np.where(known, np.where(labels[i] == labels[j], 0.99, 0.01), wp)
        s = (p[i] * p[j]).sum(1)
        for src, w in (("pred", np.clip(wp, EPS, 1 - EPS)), ("true", wt)):
            preds[f"potts_b2_k{k}_{src}"] = mean_field(logp, i, j, w, 2.0)
            th = {"cc": logit(w), "ccj": logit(w) + logit(s)}
            for name, t in th.items():
                g0 = gaec(n, i, j, t)
                parts[f"{name}_gaec_k{k}_{src}"] = g0
                parts[f"{name}_gaec_km_k{k}_{src}"] = node_moves(n, i, j, t, g0)
                if k == 24 and src == "pred":  # post hoc bias sensitivity
                    for wstar in (0.3, 0.7):
                        tb = t - np.log(wstar / (1 - wstar))
                        parts[f"{name}_gaec_km_k{k}_{src}_posthoc_w{wstar}"] = node_moves(n, i, j, tb, gaec(n, i, j, tb))
            try:
                parts[f"spec_k{k}_{src}"] = spectral(n, i, j, w, k_spec)
            except Exception as ex:  # noqa: BLE001
                print("spectral failed", it["window_id"], k, src, repr(ex)[:120], flush=True)
                parts[f"spec_k{k}_{src}"] = np.arange(n)
                fails += 1
    for name, g in parts.items():
        preds[f"{name}_free"] = free_match(logp, g)
        preds[f"{name}_hung"] = match_groups(logp, g)
    rows = []
    base = {key: it[key] for key in ("sequence", "H", "window_id", "n_anchors")}
    for name, a in preds.items():
        rows.append({**base, "method": name, "accuracy": float(np.mean(a[r] == truth)),
                     "pair_f1": pair_f1(truth, a[r])[2], "n_clusters": np.nan, "partition_f1": np.nan})
    for name, g in parts.items():
        for m in ("free", "hung"):
            rows[[x["method"] for x in rows].index(f"{name}_{m}")].update(
                n_clusters=len(np.unique(g)), partition_f1=pair_f1(truth, g[r])[2])
    o = match_groups(logp[r], truth)
    rows.append({**base, "method": "oracle_partition", "accuracy": float(np.mean(o == truth)),
                 "pair_f1": pair_f1(truth, o)[2], "n_clusters": len(np.unique(truth)), "partition_f1": 1.0})
    for x in rows:
        x.update(n_targets=n, k_spec=k_spec, spec_fail=fails)
    return rows


def decode(items):
    rows = []
    for c, it in enumerate(items):
        rows += decode_window(it)
        print(c + 1, "/", len(items), it["sequence"], it["H"], len(it["p"]), flush=True)
    df = pd.DataFrame(rows).assign(dense=lambda x: x.n_anchors > 200)
    df.to_parquet(RESULTS / "global_partition_decode.parquet", index=False)
    return df


def summary(df=None):
    if df is None:
        df = pd.read_parquet(RESULTS / "global_partition_decode.parquet")
    pd.set_option("display.width", 250)
    print("windows", df.drop_duplicates(["H", "window_id", "sequence"]).groupby(["H", "dense"]).size().to_dict())
    for col in ("accuracy", "pair_f1", "partition_f1", "n_clusters"):
        t = pd.concat([df.pivot_table(index="method", columns="H", values=col),
                       df[df.dense].pivot_table(index="method", columns="H", values=col)], axis=1, keys=["all", "dense"])
        print(f"\n== {col} (window mean)\n" + t.round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary()
    elif sys.argv[1:] == ["decode"]:
        with open(CACHE, "rb") as f:
            summary(decode(pickle.load(f)))
    else:
        summary(decode(collect()))
