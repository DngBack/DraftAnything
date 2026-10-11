"""Optical-flow baseline: pretrained RAFT between the two snapshots + nearest assignment (docs/flow_baseline_protocol.md).

Grid of 16 settings chosen on M0-M2, M3/M4 evaluated once with the selected setting (all settings are logged).

  python experiments/flow_baseline.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE, examples  # noqa: E402
from experiments.lineage_glue import INDEX  # noqa: E402

HS = (33, 66, 132)
GRID = list(itertools.product(("large", "small"), (12, 32), (1,)))  # scale 2 dropped (protocol amendment)  # direction is free: both flows are computed


def model_of(name, dev):
    from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small
    m = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2) if name == "large" else raft_small(weights=Raft_Small_Weights.C_T_V2)
    return m.to(dev).eval()


def prep(im, scale, dev):
    lo, hi = np.percentile(im, [1, 99.5])
    x = np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 2 - 1  # RAFT expects [-1, 1]
    t = torch.from_numpy(x.astype(np.float32))[None, None].repeat(1, 3, 1, 1).to(dev)
    if scale != 1:
        t = F.interpolate(t, scale_factor=scale, mode="bilinear", align_corners=False)
    h, w = t.shape[-2:]
    return F.pad(t, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")


def sample(flow, pos, scale):
    """flow (2, H, W) in scaled pixels (x, y); pos (n, 2) as (y, x) in original pixels -> displacement (n, 2) as (y, x)."""
    h, w = flow.shape[-2:]
    g = torch.as_tensor(pos[:, ::-1] * scale, dtype=torch.float32, device=flow.device)
    g = torch.stack([g[:, 0] / (w - 1) * 2 - 1, g[:, 1] / (h - 1) * 2 - 1], -1)[None, None]
    v = F.grid_sample(flow[None], g, mode="bilinear", align_corners=True)[0, :, 0].T.cpu().numpy()
    return v[:, ::-1] / scale


def keep(e):
    tg = e["targets"]
    ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
    ro = {int(v): k for k, v in enumerate(e["target"].local_detection_id)}
    co = {int(v): k for k, v in enumerate(e["anchor"].local_detection_id)}
    ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
    if not len(ok):
        return None
    return ok.target_detection_id.map(ro).to_numpy(), ok.ancestor_detection_id.map(co).to_numpy()


@torch.no_grad()
def run(aliases, settings, dev="cuda"):
    windows = Windows(BENCH, ROOT / "data").windows
    models = {}
    rows = []
    for alias in aliases:
        paths = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").image_path
        for h in HS:
            for e in examples(alias, windows, compensate=False, horizon=h):
                k = keep(e)
                if k is None:
                    continue
                r, truth = k
                pa = e["anchor"][["center_y", "center_x"]].to_numpy(np.float64)
                pb = e["target"][["center_y", "center_x"]].to_numpy(np.float64)
                ia = tifffile.imread(paths[e["window"]["anchor_frame"]])
                ib = tifffile.imread(paths[e["window"]["target_frame"]])
                base = {"sequence": alias, "H": h, "window_id": e["window"]["window_id"], "n_anchors": len(pa)}
                near = cKDTree(pa).query(pb)[1]
                rows.append({**base, "method": "nearest", "accuracy": float(np.mean(near[r] == truth))})
                for name, iters, scale in settings:
                    if name not in models:
                        models[name] = model_of(name, dev)
                    ta, tb = prep(ia, scale, dev), prep(ib, scale, dev)
                    fb = models[name](tb, ta, num_flow_updates=iters)[-1][0]  # later -> earlier
                    ff = models[name](ta, tb, num_flow_updates=iters)[-1][0]  # earlier -> later
                    back = cKDTree(pa).query(pb + sample(fb, pb, scale))[1]
                    fwd = cKDTree(pa + sample(ff, pa, scale)).query(pb)[1]
                    for d, g in (("backward", back), ("forward", fwd)):
                        rows.append({**base, "method": f"raft_{name}_{d}_it{iters}_x{scale}",
                                     "accuracy": float(np.mean(g[r] == truth))})
            print(alias, h, flush=True)
    return pd.DataFrame(rows)


def main():
    dev = run(("M0", "M1", "M2"), GRID)
    dev.to_parquet(RESULTS / "flow_baseline_dev.parquet", index=False)
    score = dev.groupby(["method", "H"]).accuracy.mean().groupby("method").mean().sort_values(ascending=False)
    print(score.round(4).to_string())
    pick = score.drop("nearest").idxmax()
    print("selected:", pick, flush=True)
    _, name, d, it, sc = pick.split("_")
    test = run(("M3", "M4"), [(name, int(it[2:]), int(sc[1:]))])
    test = test[test.method.isin(["nearest", pick])]
    test.to_parquet(RESULTS / "flow_baseline.parquet", index=False)
    test["dense"] = test.n_anchors > 200
    print(test.pivot_table(index="method", columns="H", values="accuracy").round(4).to_string())
    print(test[test.dense].pivot_table(index="method", columns="H", values="accuracy").round(4).to_string())


if __name__ == "__main__":
    main()
