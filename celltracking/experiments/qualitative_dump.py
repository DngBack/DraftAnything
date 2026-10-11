"""Dump per-target predictions of one window for the qualitative paper figure.

  python experiments/qualitative_dump.py --window M4_a0604_b0670_ends --tag raft_v3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.image_partition_decode import image_affinity, mean_field  # noqa: E402
from experiments.lineage_glue import INDEX, frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.snapshot_genealogy import CKPT, Net  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="M4_a0604_b0670_ends")
    ap.add_argument("--tag", default="raft_v3")
    a = ap.parse_args()
    dev = "cuda"
    alias = a.window.split("_")[0]
    h = int(a.window.split("_b")[1][:4]) - int(a.window.split("_a")[1][:4])
    model, margs = load(a.tag)
    ck = torch.load(CKPT / "snapshot_genealogy.pt", map_location=dev, weights_only=False)
    net = Net(True).to(dev)
    net.load_state_dict(ck["image"])
    net.eval()
    mt, st = torch.as_tensor(ck["mean"], device=dev), torch.as_tensor(ck["std"], device=dev)
    windows = Windows(BENCH, ROOT / "data").windows
    e = next(x for x in examples(alias, windows, compensate=False, horizon=h) if x["window"]["window_id"] == a.window)
    anc, tgt = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
    prob, _, _ = predict(model, frame_dict(anc), frame_dict(tgt), h, image_shape_of(alias),
                         margs.get("no_shape", False), init=margs.get("init", "radial"))
    p = prob[:, :-1] / np.maximum(prob[:, :-1].sum(1, keepdims=True), 1e-9)
    logp = np.log(np.maximum(p, 1e-9))
    paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
    raw = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
    im = (raw - raw.mean()) / (raw.std() + 1e-6)
    i, j, w = image_affinity(net, mt, st, im, tgt, h, dev)
    tg = e["targets"]
    ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
    ro = {int(v): k for k, v in enumerate(tgt.local_detection_id)}
    co = {int(v): k for k, v in enumerate(anc.local_detection_id)}
    ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
    pa, pb = anc[["center_y", "center_x"]].to_numpy(), tgt[["center_y", "center_x"]].to_numpy()
    out = {"scored": ok.target_detection_id.map(ro).to_numpy(), "truth": ok.ancestor_detection_id.map(co).to_numpy(),
           "anchor_pos": pa, "target_pos": pb, "target_image": raw,
           "anchor_image": tifffile.imread(paths[e["window"]["anchor_frame"]]).astype(np.float32),
           "nearest": cKDTree(pa).query(pb)[1], "ours": p.argmax(1), "ours_potts": mean_field(logp, i, j, w, 2.0)}
    for k in ("nearest", "ours", "ours_potts"):
        print(k, np.mean(out[k][out["scored"]] == out["truth"]))
    np.savez_compressed(RESULTS / f"qualitative_{a.window}.npz", **out)


if __name__ == "__main__":
    main()
