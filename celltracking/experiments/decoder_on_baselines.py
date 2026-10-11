"""Genealogy decoder on baseline probabilities (docs/decoder_on_baselines_protocol.md).

Applies the Potts mean-field decoder of the main text (beta = 2, 8-NN image genealogy affinities) to fine-tuned
Trackastra and to family-centre regression (GBDT), on the decoder windows of OIAM M3/M4, and compares with the
LineageRAFT decoder rows of partition_metrics_cache.pkl on the same windows. Run once.

  python experiments/decoder_on_baselines.py            # inference + cache + summary
  python experiments/decoder_on_baselines.py summary    # summary from cache
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.family import collective_assignment  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import Affinity, BENCH, RESULTS, ROOT, SEQUENCE, examples, probabilities  # noqa: E402
from experiments.image_partition_decode import image_affinity, mean_field  # noqa: E402
from experiments.lineage_glue import INDEX  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402
from experiments.partition_metrics import SEEDS, genealogy  # noqa: E402
from experiments.trackastra_finetune import CKPT as TRA_CKPT, load_frames  # noqa: E402

HS = (33, 66, 132)
TGRID = np.round(np.exp(np.linspace(np.log(0.5), np.log(8.0), 25)), 4)
CACHE = RESULTS / "decoder_on_baselines_cache.pkl"
FLOOR = 1e-9


def keep(e):
    tg = e["targets"]
    ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
    ro = {int(v): k for k, v in enumerate(e["target"].local_detection_id)}
    co = {int(v): k for k, v in enumerate(e["anchor"].local_detection_id)}
    ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
    if not len(ok) or len(co) < 2 or len(e["target"]) < 9:
        return None
    return ok.target_detection_id.map(ro).to_numpy(), ok.ancestor_detection_id.map(co).to_numpy()


def trackastra_prob(model, alias, e):
    """Row-normalised association weights target x anchor; unlinked targets -> their nearest anchor."""
    w = e["window"]
    imgs, masks = load_frames(alias, w["anchor_frame"], w["target_frame"])
    pred = model._predict(imgs, masks)
    anc, tgt = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
    ai = {int(t): k for k, t in enumerate(anc.gt_track_id)}
    ti = {int(t): k for k, t in enumerate(tgt.gt_track_id)}
    node = {n["id"]: (int(n["time"]), int(n["label"])) for n in pred["nodes"]}
    p = np.zeros((len(tgt), len(anc)), np.float64)
    for (u, v), wt in pred["weights"]:
        (t0, l0), (t1, l1) = sorted([node[int(u)], node[int(v)]])
        if t0 == 0 and t1 == 1 and l0 in ai and l1 in ti:
            p[ti[l1], ai[l0]] = max(p[ti[l1], ai[l0]], float(wt))
    near = cKDTree(anc[["center_y", "center_x"]].to_numpy()).query(tgt[["center_y", "center_x"]].to_numpy())[1]
    empty = p.sum(1) == 0
    p[empty, near[empty]] = 1.0
    return (p / p.sum(1, keepdims=True)).astype(np.float32), float(empty.mean())


def collect():
    from trackastra.model import Trackastra
    dev = "cuda"
    net, mt, st = genealogy("snapshot_genealogy.pt", dev)
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    affinity = Affinity()
    affinity.load_state_dict(saved["model"])
    affinity.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    gbdt = fit_multi([e for alias in ("M0", "M1", "M2") for h in HS
                      for e in examples(alias, windows, compensate=False, horizon=h)])
    out = []
    for h in HS:
        model = Trackastra.from_pretrained("general_2d", device=dev)
        ck = torch.load(TRA_CKPT / f"h{h}" / "best.pt", map_location=dev)
        model.transformer.load_state_dict(ck["model"])
        model.transformer.eval()
        print(f"H={h}: Trackastra checkpoint step {ck['step']}", flush=True)
        for alias in ("M2", "M3", "M4"):
            test = alias != "M2"
            items = examples(alias, windows, compensate=False, horizon=h)
            if test:
                probabilities(affinity, saved["mean"], saved["std"], items)
                paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
            for e in items:
                k = keep(e)
                if k is None:
                    continue
                r, truth = k
                p_tra, unlinked = trackastra_prob(model, alias, e)
                item = {"sequence": alias, "H": h, "window_id": e["window"]["window_id"], "n_anchors": len(e["anchor"]),
                        "r": r, "truth": truth, "p_tra": p_tra, "unlinked": unlinked}
                if test:
                    cost = multi_cost(e, gbdt)
                    item["gbdt_cost"] = cost.astype(np.float32)
                    item["gbdt_own"] = np.asarray(collective_assignment(cost, e["edges"], e["prob"], **decoder))
                    im = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
                    im = (im - im.mean()) / (im.std() + 1e-6)
                    item["edges"] = image_affinity(net, mt, st, im, e["target"].reset_index(drop=True), h, dev)
                else:  # temperature fitting needs only the scored rows
                    item["p_tra"] = p_tra[r]
                out.append(item)
            print(alias, h, flush=True)
        del model
        torch.cuda.empty_cache()
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def fit_t(items, h):
    val = [it for it in items if it["sequence"] == "M2" and it["H"] == h]
    nll = []
    for t in TGRID:
        s = []
        for it in val:
            lp = log_softmax(np.log(np.maximum(it["p_tra"].astype(np.float64), FLOOR)) / t)
            s.append(-lp[np.arange(len(it["truth"])), it["truth"]])
        nll.append(np.concatenate(s).mean())
    return float(TGRID[int(np.argmin(nll))])


def paired_ci(a, b, n=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def summary(items=None):
    if items is None:
        with open(CACHE, "rb") as f:
            items = pickle.load(f)
    temps = {h: fit_t(items, h) for h in HS}
    print("fitted T (M2 NLL):", temps)
    with open(RESULTS / "partition_metrics_cache.pkl", "rb") as f:
        raft = {(it["H"], it["window_id"]): it for it in pickle.load(f)}
    rows = []
    for it in items:
        if it["sequence"] == "M2":
            continue
        r, truth, h = it["r"], it["truth"], it["H"]
        lp_tra = np.log(np.maximum(it["p_tra"].astype(np.float64), FLOOR))
        lp_gb = log_softmax(-it["gbdt_cost"].astype(np.float64) / 4.0)
        pred = {"trackastra_argmax": lp_tra.argmax(1),
                "trackastra_mf2_T1": mean_field(lp_tra, *it["edges"], 2.0),
                "trackastra_mf2": mean_field(log_softmax(lp_tra / temps[h]), *it["edges"], 2.0),
                "gbdt_own": it["gbdt_own"], "gbdt_argmax": lp_gb.argmax(1),
                "gbdt_mf2": mean_field(lp_gb, *it["edges"], 2.0)}
        acc = {k: float(np.mean(v[r] == truth)) for k, v in pred.items()}
        ri = raft.get((h, it["window_id"]))
        if ri is not None:
            assert np.array_equal(ri["truth"], truth), it["window_id"]
            for name in ("argmax", "mf2"):
                acc[f"raft_{name}"] = float(np.mean([np.mean(ri["pred"][f"{t}_{name}"] == truth) for t in SEEDS]))
            acc["raft_ens_mf2"] = float(np.mean(ri["pred"]["ens_mf2"] == truth))
        for k, v in acc.items():
            rows.append({"sequence": it["sequence"], "H": h, "window_id": it["window_id"], "n_anchors": it["n_anchors"],
                         "unlinked": it["unlinked"], "method": k, "accuracy": v})
    df = pd.DataFrame(rows).assign(dense=lambda x: x.n_anchors > 200)
    df.to_parquet(RESULTS / "decoder_on_baselines.parquet", index=False)
    pd.set_option("display.width", 250)
    for lab, sub in (("all", df), ("dense", df[df.dense])):
        print(lab)
        print(sub.pivot_table(index="method", columns="H", values="accuracy").round(3).to_string())
    w = df.pivot_table(index=["H", "window_id"], columns="method", values="accuracy").dropna()
    print("windows per H:", w.groupby(level=0).size().to_dict(), "unlinked frac:",
          df.groupby("H").unlinked.mean().round(4).to_dict())
    tests = [("G1", "raft_mf2", "trackastra_mf2"), ("G2", "raft_mf2", "gbdt_mf2"),
             ("G3a", "raft_ens_mf2", "trackastra_mf2"), ("G3b", "raft_ens_mf2", "gbdt_mf2"),
             ("gain tra", "trackastra_mf2", "trackastra_argmax"), ("gain gbdt", "gbdt_mf2", "gbdt_own"),
             ("gain raft", "raft_mf2", "raft_argmax")]
    for name, a, b in tests:
        print(name, a, "-", b, {h: tuple(round(x, 4) for x in paired_ci(w.loc[h][a], w.loc[h][b])) for h in HS})


if __name__ == "__main__":
    summary(None if sys.argv[1:] == ["summary"] else collect())
