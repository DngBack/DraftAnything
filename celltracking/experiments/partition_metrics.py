"""Family-partition metrics on OIAM M3/M4: does a method group descendants correctly?

Ancestor accuracy asks whether each target gets the right ancestor; the partition metrics ignore which ancestor
a group is given and score only the grouping of scored targets into families. Pairwise family F1 over all pairs of
scored targets in a window (a pair is positive if both targets share the true ancestor), and the adjusted Rand
index. Windows with at least nine targets (as for the decoder), every method on the same windows. The oracle
partition is not listed: it is the true partition by construction (F1 = ARI = 1).

  python experiments/partition_metrics.py            # inference + cache + summary
  python experiments/partition_metrics.py summary    # summary from cache
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
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.family import collective_assignment  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import Affinity, BENCH, RESULTS, ROOT, SEQUENCE, examples, probabilities  # noqa: E402
from experiments.image_partition_decode import image_affinity, mean_field  # noqa: E402
from experiments.lineage_glue import INDEX, frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402
from experiments.snapshot_genealogy import CKPT, Net  # noqa: E402

SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
PSEUDO = ("raft_v3p", "raft_v3p_s1", "raft_v3p_s2")
HS = (33, 66, 132)
CACHE = RESULTS / "partition_metrics_cache.pkl"


def genealogy(name, dev):
    ck = torch.load(CKPT / name, map_location=dev, weights_only=False)
    net = Net(True).to(dev)
    net.load_state_dict(ck["image"])
    net.eval()
    return net, torch.as_tensor(ck["mean"], device=dev), torch.as_tensor(ck["std"], device=dev)


def normalise(p):
    p = p[:, :-1]
    return p / np.maximum(p.sum(1, keepdims=True), 1e-9)


def collect():
    dev = "cuda"
    models = {t: load(t) for t in SEEDS + PSEUDO}
    gens = {"real": genealogy("snapshot_genealogy.pt", dev), "pseudo": genealogy("snapshot_genealogy_pseudo.pt", dev)}
    decoder = json.loads((RESULTS / "family_set_h132_raw_config.json").read_text())
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    affinity = Affinity()
    affinity.load_state_dict(saved["model"])
    affinity.eval()
    windows = Windows(BENCH, ROOT / "data").windows
    gbdt = fit_multi([e for alias in ("M0", "M1", "M2") for h in HS
                      for e in examples(alias, windows, compensate=False, horizon=h)])
    out = []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
        for h in HS:
            items = examples(alias, windows, compensate=False, horizon=h)
            probabilities(affinity, saved["mean"], saved["std"], items)
            for e in items:
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): k for k, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): k for k, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok) or len(co) < 2 or len(e["target"]) < 9:
                    continue
                r = ok.target_detection_id.map(ro).to_numpy()
                fa, fb = frame_dict(e["anchor"]), frame_dict(e["target"])
                raw = {t: predict(m, fa, fb, h, shape, a.get("no_shape", False), init=a.get("init", "radial"))[0]
                       for t, (m, a) in models.items()}
                im = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
                im = (im - im.mean()) / (im.std() + 1e-6)
                df = e["target"].reset_index(drop=True)
                edges = {k: image_affinity(net, mt, st, im, df, h, dev) for k, (net, mt, st) in gens.items()}
                pred = {"nearest": cKDTree(fa["pos"]).query(fb["pos"])[1],
                        "gbdt": np.asarray(collective_assignment(multi_cost(e, gbdt), e["edges"], e["prob"], **decoder))}
                for t in SEEDS:
                    p = normalise(raw[t])
                    pred[f"{t}_argmax"] = p.argmax(1)
                    pred[f"{t}_mf2"] = mean_field(np.log(np.maximum(p, 1e-9)), *edges["real"], 2.0)
                for name, tags, gen in (("ens", SEEDS, "real"), ("ens_noann", PSEUDO, "pseudo")):
                    p = normalise(np.mean([raw[t] for t in tags], 0))
                    pred[f"{name}_argmax"] = p.argmax(1)
                    pred[f"{name}_mf2"] = mean_field(np.log(np.maximum(p, 1e-9)), *edges[gen], 2.0)
                out.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"], "n_anchors": len(co),
                            "truth": ok.ancestor_detection_id.map(co).to_numpy(),
                            "pred": {k: v[r] for k, v in pred.items()}})
            print(alias, h, flush=True)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def pair_f1(truth, pred):
    c = pd.crosstab(truth, pred).to_numpy()
    tp = (c * (c - 1) / 2).sum()
    true_pairs = (c.sum(1) * (c.sum(1) - 1) / 2).sum()
    pred_pairs = (c.sum(0) * (c.sum(0) - 1) / 2).sum()
    if true_pairs == 0:
        return np.nan, np.nan, np.nan
    p = tp / pred_pairs if pred_pairs else 0.0
    rcl = tp / true_pairs
    return p, rcl, (2 * p * rcl / (p + rcl) if p + rcl else 0.0)


def summary(items=None):
    if items is None:
        with open(CACHE, "rb") as f:
            items = pickle.load(f)
    rows = []
    for it in items:
        for name, g in it["pred"].items():
            p, rc, f1 = pair_f1(it["truth"], g)
            rows.append({"sequence": it["sequence"], "H": it["H"], "window_id": it["window_id"],
                         "n_anchors": it["n_anchors"], "method": name, "accuracy": float(np.mean(g == it["truth"])),
                         "pair_precision": p, "pair_recall": rc, "pair_f1": f1,
                         "ari": adjusted_rand_score(it["truth"], g)})
    df = pd.DataFrame(rows).assign(dense=lambda x: x.n_anchors > 200)
    for t in ("argmax", "mf2"):  # three-seed means of single models
        sub = df[df.method.isin([f"{s}_{t}" for s in SEEDS])]
        mean = sub.groupby(["sequence", "H", "window_id", "n_anchors", "dense"], as_index=False).mean(numeric_only=True)
        df = pd.concat([df, mean.assign(method=f"single_{t}")], ignore_index=True)
    df.to_parquet(RESULTS / "partition_metrics.parquet", index=False)
    order = ["nearest", "gbdt", "single_argmax", "single_mf2", "ens_argmax", "ens_mf2", "ens_noann_mf2"]
    for lab, sub in (("all", df), ("dense", df[df.dense])):
        for col in ("accuracy", "pair_f1", "ari", "pair_precision", "pair_recall"):
            print(lab, col)
            print(sub[sub.method.isin(order)].pivot_table(index="method", columns="H", values=col)
                  .reindex(order).round(3).to_string())


if __name__ == "__main__":
    summary() if sys.argv[1:] == ["summary"] else summary(collect())
