"""Self-labelled Trackastra vs self-labelled LineageRAFT, argmax and decoded (docs/trackastra_pseudo_protocol.md).

Decoded rows: decoder windows of OIAM M3/M4 (>= 9 targets, >= 2 anchors), Potts mean field (beta = 2, 10 iterations,
8-NN target graph) on the affinities of the annotation-free genealogy net snapshot_genealogy_pseudo.pt, unaries
log p with p = Trackastra association weights (T = 1, decoder_on_baselines.trackastra_prob) or the probabilities of
one self-labelled LineageRAFT seed. Argmax rows (U1, U3): the eval parquets on all windows of Tab. 2. Run once.

  python experiments/trackastra_pseudo_decode.py            # inference + cache + summary
  python experiments/trackastra_pseudo_decode.py summary    # summary from cache
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.block_bootstrap import ci  # noqa: E402
from experiments.decoder_on_baselines import keep, trackastra_prob  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.image_partition_decode import image_affinity, mean_field  # noqa: E402
from experiments.lineage_glue import INDEX, frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.snapshot_genealogy import CKPT as GEN_CKPT, Net  # noqa: E402
from experiments.trackastra_finetune import CKPT as TRA_CKPT  # noqa: E402

HS = (33, 66, 132)
RAFT = ("raft_pseudo", "raft_pseudo_s1", "raft_pseudo_s2")
# checkpoint dirs: per-gap models at 33/66, the over-gap model at 132 (as in Tab. 2)
TRA = {"tra_pseudo": {33: "h33_pseudo", 66: "h66_pseudo", 132: "h132_multi_pseudo"},
       "tra_m012": {33: "h33_m012", 66: "h66_m012", 132: "h132_multi_m012"}}
CACHE = RESULTS / "trackastra_pseudo_decode_cache.pkl"
FLOOR = 1e-9


def collect():
    from trackastra.model import Trackastra
    dev = "cuda"
    ck = torch.load(GEN_CKPT / "snapshot_genealogy_pseudo.pt", map_location=dev, weights_only=False)
    net = Net(True).to(dev)
    net.load_state_dict(ck["image"])
    net.eval()
    mt, st = torch.as_tensor(ck["mean"], device=dev), torch.as_tensor(ck["std"], device=dev)
    raft = [load(t) for t in RAFT]
    windows = Windows(BENCH, ROOT / "data").windows
    out = []
    for h in HS:
        tra = {}
        for name, dirs in TRA.items():
            m = Trackastra.from_pretrained("general_2d", device=dev)
            m.transformer.load_state_dict(torch.load(TRA_CKPT / dirs[h] / "best.pt", map_location=dev)["model"])
            m.transformer.eval()
            tra[name] = m
        for alias in ("M3", "M4"):
            shape = image_shape_of(alias)
            paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
            for e in examples(alias, windows, compensate=False, horizon=h):
                k = keep(e)
                if k is None:
                    continue
                r, truth = k
                item = {"sequence": alias, "H": h, "window_id": e["window"]["window_id"], "n_anchors": len(e["anchor"]),
                        "r": r, "truth": truth, "p": {}, "unlinked": {}}
                for name, m in tra.items():
                    item["p"][name], item["unlinked"][name] = trackastra_prob(m, alias, e)
                for t, (m, ma) in zip(RAFT, raft):
                    p = predict(m, frame_dict(e["anchor"]), frame_dict(e["target"]), h, shape,
                                ma.get("no_shape", False), init=ma.get("init", "radial"))[0][:, :-1]
                    item["p"][t] = (p / np.maximum(p.sum(1, keepdims=True), FLOOR)).astype(np.float32)
                im = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
                im = (im - im.mean()) / (im.std() + 1e-6)
                item["edges"] = image_affinity(net, mt, st, im, e["target"].reset_index(drop=True), h, dev)
                out.append(item)
            print(alias, h, flush=True)
        del tra
        torch.cuda.empty_cache()
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def decoded_frame(items):
    rows = []
    for it in items:
        r, truth = it["r"], it["truth"]
        acc = {}
        for name, p in it["p"].items():
            lp = np.log(np.maximum(p.astype(np.float64), FLOOR))
            acc[f"{name}_argmax"] = float(np.mean(lp.argmax(1)[r] == truth))
            acc[f"{name}_mf2"] = float(np.mean(mean_field(lp, *it["edges"], 2.0)[r] == truth))
        for d in ("argmax", "mf2"):
            acc[f"raft_pseudo_mean_{d}"] = float(np.mean([acc[f"{t}_{d}"] for t in RAFT]))
        for k, v in acc.items():
            rows.append({"sequence": it["sequence"], "H": it["H"], "window_id": it["window_id"],
                         "n_anchors": it["n_anchors"], "method": k, "accuracy": v})
    return pd.DataFrame(rows)


def argmax_frame():
    """All windows of Tab. 2: self-labelled LineageRAFT seeds, Trackastra per-gap (33/66) and over-gap (132)."""
    parts = []
    for t in RAFT:
        d = pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet")
        parts.append(d[d.method == f"{t}_argmax"].assign(method=t))
    for name, (per_gap, multi) in {"tra_pseudo": ("trackastra_ft_pseudo", "trackastra_ft_multi_pseudo"),
                                   "tra_m012": ("trackastra_ft_m012", "trackastra_ft_multi_m012")}.items():
        d = pd.concat([pd.read_parquet(RESULTS / f"{per_gap}.parquet").query("H in (33, 66)"),
                       pd.read_parquet(RESULTS / f"{multi}.parquet").query("H == 132")])
        parts.append(d[d.method == "trackastra_ft_argmax"].assign(method=name))
    for t in ("raft_v3", "raft_v3_s1", "raft_v3_s2"):
        d = pd.read_parquet(RESULTS / f"raft_eval_{t}.parquet")
        parts.append(d[d.method == f"{t}_argmax"].assign(method=t))
    df = pd.concat(parts)[["sequence", "H", "window_id", "n_anchors", "method", "accuracy"]]
    w = df.pivot_table(index=["H", "window_id", "n_anchors"], columns="method", values="accuracy")
    w["raft_pseudo_mean"] = w[list(RAFT)].mean(axis=1)
    w["raft_v3_mean"] = w[["raft_v3", "raft_v3_s1", "raft_v3_s2"]].mean(axis=1)
    return w


def compare(w, a, b, name):
    rows = []
    for h in HS:
        if h not in w.index.get_level_values(0):
            continue
        x = w.loc[h].dropna(subset=[a, b]).reset_index()
        for lab, sub in (("all", x), ("dense", x[x.n_anchors > 200])):
            sub = sub.assign(seq=sub.window_id.str[:2],
                             anchor=sub.window_id.str.extract(r"_a(\d+)_", expand=False).astype(int))
            sub = sub.sort_values(["seq", "anchor"])
            dm = [(g[a] - g[b]).to_numpy() for _, g in sub.groupby("seq")]
            m, lo, hi = ci(dm, 0)
            _, blo, bhi = ci(dm, 4)
            rows.append({"test": name, "H": h, "subset": lab, "n": len(sub), "a": sub[a].mean(), "b": sub[b].mean(),
                         "diff": m, "win_lo": lo, "win_hi": hi, "blk4_lo": blo, "blk4_hi": bhi})
    return rows


def summary(items=None):
    if items is None:
        with open(CACHE, "rb") as f:
            items = pickle.load(f)
    dec = decoded_frame(items)
    dec.to_parquet(RESULTS / "trackastra_pseudo_decode.parquet", index=False)
    dw = dec.pivot_table(index=["H", "window_id", "n_anchors"], columns="method", values="accuracy")
    aw = argmax_frame()
    pd.set_option("display.width", 250)
    print("argmax, all windows of Tab. 2\n", aw.groupby(level=0).mean().round(4).T.to_string())
    print("decoder windows\n", dw.groupby(level=0).mean().round(4).T.to_string())
    print("unlinked targets (Trackastra):",
          {n: {h: round(float(np.mean([it["unlinked"][n] for it in items if it["H"] == h])), 4) for h in HS} for n in TRA})
    rows = (compare(aw, "raft_pseudo_mean", "tra_pseudo", "U1 argmax")
            + compare(dw, "raft_pseudo_mean_mf2", "tra_pseudo_mf2", "U2 decoded")
            + compare(aw, "tra_m012", "tra_pseudo", "cost tra (annot - self)")
            + compare(aw, "raft_v3_mean", "raft_pseudo_mean", "cost raft (annot - self)")
            + compare(dw, "tra_pseudo_mf2", "tra_pseudo_argmax", "decoder gain tra")
            + compare(dw, "raft_pseudo_mean_mf2", "raft_pseudo_mean_argmax", "decoder gain raft"))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "trackastra_pseudo_tests.csv", index=False)
    print(res.round(4).to_string(index=False))
    a = res[res.subset == "all"].set_index(["test", "H"])
    u1 = all(a.loc[("U1 argmax", h), "win_lo"] > 0 for h in (33, 66))
    u2 = all(a.loc[("U2 decoded", h), "win_lo"] > 0 for h in (33, 66))
    u3 = all(a.loc[("cost tra (annot - self)", h), "diff"] <= 0.05 for h in (33, 66))
    print(f"U1 holds: {u1}\nU2 holds: {u2}\nU3 holds: {u3}")


if __name__ == "__main__":
    summary(None if sys.argv[1:] == ["summary"] else collect())
