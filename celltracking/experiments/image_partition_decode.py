"""Partition-first decoding: cluster target cells by single-frame image genealogy,
then assign each cluster to one anchor with LineageRAFT probabilities.

The snapshot-genealogy network (snapshot_genealogy.py) scores kNN pairs in the
target frame for "same ancestor H minutes ago" (tau < H). Average linkage on
those scores gives a family partition; clusters are matched to anchors either
one-to-one first (Hungarian, anchors reused only after all are used) or freely
(argmax of summed log-probability). No lineage labels are used at inference.
Development analysis on OIAM M3/M4 (viewed before).

  python experiments/image_partition_decode.py --tag raft_v2_darcy_30k
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.family import family_edges  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.lineage_glue import INDEX, frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.raft_family_probe import average_linkage, match_groups  # noqa: E402
from experiments.snapshot_genealogy import CKPT, THRESHOLDS, Net, patches  # noqa: E402


@torch.no_grad()
def image_affinity(net, mt, st, img, df, h, dev, make=patches):
    e, x = family_edges(df, 8)
    pos = df[["center_y", "center_x"]].to_numpy(np.float32)
    k = THRESHOLDS.index(h)
    out = []
    for s in range(0, len(e), 512):
        ek = e[s:s + 512]
        xg = (torch.as_tensor(x[s:s + 512], device=dev) - mt) / st
        z = 0.5 * (net(xg, make(img, pos, ek, dev)).float() +
                   net(xg, make(img, pos, ek[:, ::-1].copy(), dev)).float())
        out.append(torch.sigmoid(z[:, k]).cpu().numpy())
    return e[:, 0], e[:, 1], np.concatenate(out) if out else np.zeros(0)


def free_match(logp, groups):
    out = np.empty(len(groups), int)
    for g in np.unique(groups):
        m = groups == g
        out[m] = logp[m].sum(0).argmax()
    return out


def mean_field(logp, i, j, w, beta, iters=10):
    """Potts mean field: q_i ~ p_i * exp(beta * sum_j (2 w_ij - 1) q_j) on image kNN edges."""
    q = np.exp(logp - logp.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    c = 2 * w - 1
    for _ in range(iters):
        m = np.zeros_like(q)
        np.add.at(m, i, c[:, None] * q[j])
        np.add.at(m, j, c[:, None] * q[i])
        z = logp + beta * m
        q = np.exp(z - z.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
    return q.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="raft_v2_darcy_30k", help="comma-separated tags average their probabilities")
    ap.add_argument("--genealogy", choices=["real", "sim", "pseudo"], default="real",
                    help="sim: genealogy net trained on rendered simulated colonies only")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--gen-ckpt", default="", help="sim genealogy checkpoint name (e.g. sim_genealogy_v2_v2)")
    a = ap.parse_args()
    dev = "cuda"
    models = [load(t) for t in a.tag.split(",")]
    make = patches
    ck_name = "snapshot_genealogy.pt"
    if a.genealogy == "sim":
        from experiments.sim_genealogy import norm_patches
        make, ck_name = norm_patches, f"{a.gen_ckpt or 'sim_genealogy'}.pt"
    elif a.genealogy == "pseudo":  # trained on chained pseudo-lineage, no annotation
        ck_name = "snapshot_genealogy_pseudo.pt"
    ck = torch.load(CKPT / ck_name, map_location=dev, weights_only=False)
    net = Net(True).to(dev)
    net.load_state_dict(ck["image"])
    net.eval()
    mt, st = torch.as_tensor(ck["mean"], device=dev), torch.as_tensor(ck["std"], device=dev)
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
        for h in (33, 66, 132):
            for e in examples(alias, windows, compensate=False, horizon=h):
                prob = np.mean([predict(m, frame_dict(e["anchor"]), frame_dict(e["target"]), h, shape,
                                        ma.get("no_shape", False), init=ma.get("init", "radial"))[0]
                                for m, ma in models], 0)
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): k for k, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): k for k, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok) or len(co) < 2 or len(e["target"]) < 9:
                    continue
                r = ok.target_detection_id.map(ro).to_numpy()
                truth = ok.ancestor_detection_id.map(co).to_numpy()
                p = prob[:, :-1] / np.maximum(prob[:, :-1].sum(1, keepdims=True), 1e-9)
                logp = np.log(np.maximum(p, 1e-9))
                im = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
                im = (im - im.mean()) / (im.std() + 1e-6)
                i, j, w = image_affinity(net, mt, st, im, e["target"].reset_index(drop=True), h, dev, make)
                row = {"sequence": alias, "H": h, "n_anchors": len(co), "window_id": e["window"]["window_id"],
                       "argmax": float(np.mean(p[r].argmax(1) == truth)),
                       "oracle_partition": float(np.mean(match_groups(logp[r], truth) == truth))}
                for tau in a.taus:
                    g = average_linkage(len(p), i, j, w, tau)
                    row[f"img{tau}_hung"] = float(np.mean(match_groups(logp, g)[r] == truth))
                    row[f"img{tau}_free"] = float(np.mean(free_match(logp, g)[r] == truth))
                for beta in (1.0, 2.0, 4.0):
                    row[f"mf_img{beta:g}"] = float(np.mean(mean_field(logp, i, j, w, beta)[r] == truth))
                rows.append(row)
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"image_partition_decode_{a.tag.replace(',', '+')}{'_simgen' if a.genealogy == 'sim' else '_pseudogen' if a.genealogy == 'pseudo' else ''}{'_' + a.gen_ckpt if a.gen_ckpt else ''}.parquet", index=False)
    df["dense"] = df.n_anchors > 200
    cols = [c for c in df.columns if c not in ("sequence", "H", "n_anchors", "dense", "window_id")]
    print(df.groupby("H")[cols].mean().round(4).T.to_string())
    print("dense\n", df[df.dense].groupby("H")[cols].mean().round(4).T.to_string())


if __name__ == "__main__":
    main()
