"""Wall-clock time per window on OIAM M3/M4 (the windows of tab:oiam).

Times each stage of the pipeline and the OT baselines on the same windows: Darcy initialisation,
one LineageRAFT forward pass (argmax), the genealogy network on the target image, and the Potts
mean-field decoder; nearest assignment, exact capacity OT and unbalanced OT for comparison.
GPU stages are synchronised. Measured on a shared GPU, so numbers are upper bounds.

  python experiments/runtime_bench.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.darcy_flow_prior import advect  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.image_partition_decode import image_affinity, mean_field  # noqa: E402
from experiments.lineage_glue import INDEX, frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402
from experiments.ot_baseline import couple  # noqa: E402
from experiments.snapshot_genealogy import CKPT, Net  # noqa: E402

HS = (33, 66, 132)


def timed(fn, gpu=False):
    if gpu:
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if gpu:
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    dev = "cuda"
    model, a = load("raft_v3")
    ck = torch.load(CKPT / "snapshot_genealogy.pt", map_location=dev, weights_only=False)
    net = Net(True).to(dev)
    net.load_state_dict(ck["image"])
    net.eval()
    mt, st = torch.as_tensor(ck["mean"], device=dev), torch.as_tensor(ck["std"], device=dev)
    windows = Windows(BENCH, ROOT / "data").windows
    rows, warm = [], False
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
        for h in HS:
            for e in examples(alias, windows, compensate=False, horizon=h):
                if len(e["target"]) < 9 or len(e["anchor"]) < 2:
                    continue
                fa, fb = frame_dict(e["anchor"]), frame_dict(e["target"])
                src, dst = fa["pos"], fb["pos"]
                run = lambda: predict(model, fa, fb, h, shape, a.get("no_shape", False), init=a.get("init", "radial"))
                if not warm:
                    run()
                    warm = True
                r = {"sequence": alias, "H": h, "n_anchors": len(src), "n_targets": len(dst)}
                _, r["darcy_init"] = timed(lambda: advect(src, dst, h, shape))
                (p, _, _), r["lineageraft"] = timed(run, gpu=True)
                im = tifffile.imread(paths[e["window"]["target_frame"]]).astype(np.float32)
                im = (im - im.mean()) / (im.std() + 1e-6)
                (i, j, w), r["genealogy"] = timed(
                    lambda: image_affinity(net, mt, st, im, e["target"].reset_index(drop=True), h, dev), gpu=True)
                logp = np.log(np.maximum(p[:, :-1], 1e-9))
                _, r["potts"] = timed(lambda: mean_field(logp, i, j, w, 2.0))
                _, r["nearest"] = timed(lambda: cKDTree(src).query(dst)[1])
                _, r["exact_ot"] = timed(lambda: couple(src, dst, "exact", {"c": 1.5}))
                _, r["uot"] = timed(lambda: couple(src, dst, "uot", {"eps": 0.2, "rho": 1.0}))
                rows.append(r)
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "runtime_bench.parquet", index=False)
    stages = ["darcy_init", "lineageraft", "genealogy", "potts", "nearest", "exact_ot", "uot"]
    df["dense"] = df.n_anchors > 200
    print(df.groupby("dense")[["n_anchors", "n_targets"]].median())
    print((df.groupby("dense")[stages].median() * 1000).round(1).T.rename(columns={False: "sparse ms", True: "dense ms"}))
    print((df[stages].max() * 1000).round(1).rename("max ms"))
    big = df.loc[df.n_targets.idxmax()]
    print("largest window", big.to_dict())


if __name__ == "__main__":
    main()
