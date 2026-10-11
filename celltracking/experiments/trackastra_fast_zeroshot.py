"""Off-the-shelf Trackastra zero-shot on the FAST windows of the zero-shot table (docs/trackastra_zeroshot_fast_protocol.md).

  python experiments/trackastra_fast_zeroshot.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS, ROOT  # noqa: E402
from experiments.fast_microcolony_external_test import (GAPS, HELD_OUT, MINUTES_PER_FRAME, load_colony,  # noqa: E402
                                                        make_window)
from experiments.trackastra_finetune import argmax_links  # noqa: E402

PX_PER_UM = 10.0
TOL = 5.0


class Colony:
    def __init__(self, colony):
        self.zip = zipfile.ZipFile(ROOT / "data/downloads" / f"fast_{colony}.zip")
        self.name = colony
        self.cache = {}

    def frame(self, f):
        """Image and labelled components of frame f (1-based), with component centroids (x, y) in pixels."""
        if f not in self.cache:
            im = tifffile.imread(io.BytesIO(self.zip.read(f"{self.name}/Channel_1/Frame_{f - 1:04d}.tif")))
            m = tifffile.imread(io.BytesIO(self.zip.read(f"{self.name}/Corrected_Segmentations/Frame_{f - 1:04d}.tif")))
            lab, n = ndimage.label(m > 0)
            cc = np.array(ndimage.center_of_mass(m > 0, lab, range(1, n + 1))).reshape(-1, 2)[:, ::-1]
            self.cache[f] = (im, lab, cc)
        return self.cache[f]

    def relabel(self, f, xy_um):
        """Mask whose label k+1 is the component matched to cell k (positions in micrometres, (x, y))."""
        im, lab, cc = self.frame(f)
        out = np.zeros_like(lab, dtype=np.int32)
        if not len(cc):
            return im, out, np.zeros(len(xy_um), bool)
        d, j = cKDTree(cc).query(xy_um * PX_PER_UM)
        ok = d <= TOL
        used = set()
        for k in np.argsort(d):  # closest pairs first; one component per cell
            if ok[k] and j[k] not in used:
                out[lab == j[k] + 1] = k + 1
                used.add(j[k])
            else:
                ok[k] = False
        return im, out, ok


def main():
    from trackastra.model import Trackastra
    model = Trackastra.from_pretrained("general_2d", device="cuda")
    rng = np.random.default_rng(20261002)
    rows = []
    for colony in HELD_OUT:
        frames, mapping, tracks, _ = load_colony(ROOT / "data/downloads" / f"fast_{colony}.zip")
        col = Colony(colony)
        for gap in GAPS:
            for start in sorted(frames):
                if start + gap not in frames:
                    continue
                e, valid, truth = make_window(start, gap, frames, mapping, tracks, rng)
                if not len(truth):
                    continue
                a_xy = e["anchor"][["center_x", "center_y"]].to_numpy()
                b_xy = e["target"][["center_x", "center_y"]].to_numpy()
                ia, ma, oka = col.relabel(start, a_xy)
                ib, mb, okb = col.relabel(start + gap, b_xy)
                near = cKDTree(a_xy).query(b_xy)[1]
                preds = {"nearest": near}
                if oka.any() and okb.any():
                    imgs, masks = np.stack([ia, ib]).astype(np.float32), np.stack([ma, mb])
                    pred = model._predict(imgs, masks)
                    graph, _ = model.track(imgs, masks, mode="greedy", progbar_class=lambda x, **k: x)
                    greedy = {}
                    for u, v in graph.edges:
                        du, dv = graph.nodes[u], graph.nodes[v]
                        if int(du["time"]) == 0 and int(dv["time"]) == 1:
                            greedy[int(dv["label"])] = int(du["label"])
                    for name, link in (("trackastra_argmax", argmax_links(pred)), ("trackastra_greedy", greedy)):
                        g = np.array([link.get(k + 1, near[k] + 1) - 1 if okb[k] else -1 for k in range(len(b_xy))])
                        preds[name] = g
                else:
                    preds["trackastra_argmax"] = preds["trackastra_greedy"] = np.full(len(b_xy), -1)
                for name, g in preds.items():
                    rows.append({"dataset": "FAST", "sequence": colony, "window_id": f"{start}_{start + gap}",
                                 "gap_minutes": gap * MINUTES_PER_FRAME, "n_anchors": len(a_xy), "method": name,
                                 "accuracy": float(np.mean(g[valid] == truth)),
                                 "match_rate": float((oka.mean() + okb.mean()) / 2)})
            print(colony, gap, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "trackastra_fast_zeroshot.parquet", index=False)
    print(df.pivot_table(index="method", columns="gap_minutes", values="accuracy").round(4).to_string())
    print("mean over gaps:", df.groupby(["method", "gap_minutes"]).accuracy.mean().groupby("method").mean().round(4).to_dict())
    print("match rate:", round(df.match_rate.mean(), 4))
    print(df[df.method == "trackastra_argmax"].groupby("sequence").accuracy.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
