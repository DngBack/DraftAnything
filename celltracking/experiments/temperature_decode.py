"""Temperature scaling before the genealogy decoder (protocol fixed before running).

Question: the 3-seed ensemble gains more from the Potts decoder than any single model, and is
better calibrated (calibration.py). Is calibration the reason? If so, a single model with a
fitted temperature should recover most of the ensemble's decoder gain.

Protocol: for each model (raft_v3, raft_v3_s1, raft_v3_s2, ens3) and gap H, choose T on the
grid TGRID by the NLL of the true ancestor on OIAM M0-M2 windows (labels used for selection
only, as for the OT/CPD grids); then on M3/M4 report argmax (unchanged by T), ECE, and the
Potts mean-field decoder with beta=2 (setting of the main text) on log p / T. Run once.

  python experiments/temperature_decode.py            # inference + caches + summary
  python experiments/temperature_decode.py summary    # recompute from caches
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
from experiments.calibration import ece  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.image_partition_decode import image_affinity, mean_field  # noqa: E402
from experiments.lineage_glue import INDEX, frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.snapshot_genealogy import CKPT, Net  # noqa: E402

SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
HS = (33, 66, 132)
TGRID = np.round(np.exp(np.linspace(np.log(0.5), np.log(8.0), 25)), 4)
CACHE = RESULTS / "temperature_decode_cache.pkl"


def collect():
    dev = "cuda"
    models = {t: load(t) for t in SEEDS}
    ck = torch.load(CKPT / "snapshot_genealogy.pt", map_location=dev, weights_only=False)
    net = Net(True).to(dev)
    net.load_state_dict(ck["image"])
    net.eval()
    mt, st = torch.as_tensor(ck["mean"], device=dev), torch.as_tensor(ck["std"], device=dev)
    windows = Windows(BENCH, ROOT / "data").windows
    out = []
    for alias in ("M0", "M1", "M2", "M3", "M4"):
        test = alias in ("M3", "M4")
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
                probs = {}
                for t, (m, a) in models.items():
                    p = predict(m, frame_dict(e["anchor"]), frame_dict(e["target"]), h, shape,
                                a.get("no_shape", False), init=a.get("init", "radial"))[0][:, :-1]
                    probs[t] = (p / np.maximum(p.sum(1, keepdims=True), 1e-9)).astype(np.float32)
                probs["ens3"] = np.mean([probs[t] for t in SEEDS], 0)
                item = {"sequence": alias, "H": h, "n_anchors": len(co), "window_id": e["window"]["window_id"],
                        "r": ok.target_detection_id.map(ro).to_numpy(),
                        "truth": ok.ancestor_detection_id.map(co).to_numpy(), "probs": probs}
                if not test:  # temperature fitting needs only the scored rows
                    item["probs"] = {k: v[item["r"]] for k, v in probs.items()}
                    item["r"] = np.arange(len(item["r"]))
                if test:
                    im = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
                    im = (im - im.mean()) / (im.std() + 1e-6)
                    item["edges"] = image_affinity(net, mt, st, im, e["target"].reset_index(drop=True), h, dev)
                out.append(item)
            print(alias, h, flush=True)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def nll(items, name, T):
    v = []
    for it in items:
        lp = np.log(np.maximum(it["probs"][name][it["r"]], 1e-9)) / T
        lp -= np.log(np.exp(lp - lp.max(1, keepdims=True)).sum(1, keepdims=True)) + lp.max(1, keepdims=True)
        v.append(-lp[np.arange(len(it["r"])), it["truth"]])
    return float(np.concatenate(v).mean())


def summary(items=None):
    if items is None:
        with open(CACHE, "rb") as f:
            items = pickle.load(f)
    dev_items = [it for it in items if it["sequence"] in ("M0", "M1", "M2")]
    test_items = [it for it in items if it["sequence"] in ("M3", "M4")]
    rows, temps = [], []
    for name in (*SEEDS, "ens3"):
        for h in HS:
            d = [it for it in dev_items if it["H"] == h]
            T = float(TGRID[np.argmin([nll(d, name, t) for t in TGRID])])
            temps.append({"method": name, "H": h, "T": T, "dev_nll_T1": nll(d, name, 1.0), "dev_nll_T": nll(d, name, T)})
            for it in (x for x in test_items if x["H"] == h):
                p, r, truth = it["probs"][name], it["r"], it["truth"]
                logp = np.log(np.maximum(p, 1e-9))
                i, j, w = it["edges"]
                q = np.exp((logp - logp.max(1, keepdims=True)) / T)
                q /= q.sum(1, keepdims=True)
                rows.append({"method": name, "H": h, "T": T, "sequence": it["sequence"], "window_id": it["window_id"],
                             "n_anchors": it["n_anchors"], "argmax": float(np.mean(p[r].argmax(1) == truth)),
                             "mf2": float(np.mean(mean_field(logp, i, j, w, 2.0)[r] == truth)),
                             "mf2_T": float(np.mean(mean_field(logp / T, i, j, w, 2.0)[r] == truth)),
                             "conf": q[r].max(1), "correct": q[r].argmax(1) == truth,
                             "conf1": p[r].max(1)})
    tdf = pd.DataFrame(temps)
    print(tdf.round(4).to_string(index=False))
    df = pd.DataFrame(rows)
    cal = []
    for (name, h), g in df.groupby(["method", "H"]):
        c, k, c1 = (np.concatenate(g[x].to_numpy()) for x in ("conf", "correct", "conf1"))
        cal.append({"method": name, "H": h, "ece_T1": ece(c1, k.astype(float)), "ece_T": ece(c, k.astype(float))})
    print(pd.DataFrame(cal).round(4).to_string(index=False))
    df = df.drop(columns=["conf", "correct", "conf1"]).assign(dense=lambda x: x.n_anchors > 200)
    df.to_parquet(RESULTS / "temperature_decode.parquet", index=False)
    for sub, lab in ((df, "all"), (df[df.dense], "dense")):
        print(lab)
        print(sub.groupby(["method", "H"])[["argmax", "mf2", "mf2_T"]].mean().round(4).unstack("H").to_string())


if __name__ == "__main__":
    summary() if sys.argv[1:] == ["summary"] else summary(collect())
