"""Published-tracker baseline at sparse gaps: Trackastra (general_2d, pretrained)
given only the two endpoint frames (image + GT segmentation masks).

Two decodings: Trackastra's own greedy linker (at most two children per
parent), and per-target argmax of its association weights (no child limit,
fairer at long horizons). Targets without an edge fall back to the nearest
anchor. Same OIAM windows, filters and metrics as lineage_raft.evaluate.

  python experiments/trackastra_gap_baseline.py --sequences M3 M4
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
from experiments.fast_microcolony_external_test import family_f1  # noqa: E402
from experiments.lineage_glue import INDEX  # noqa: E402


def main():
    from trackastra.model import Trackastra
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="+", default=["M3", "M4"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[33, 66, 132])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="trackastra_gap")
    a = ap.parse_args()
    model = Trackastra.from_pretrained("general_2d", device=a.device)
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for alias in a.sequences:
        fr = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index")
        for h in a.horizons:
            for e in examples(alias, windows, compensate=False, horizon=h):
                w = e["window"]
                ta, tb = w["anchor_frame"], w["target_frame"]
                imgs = np.stack([tifffile.imread(fr.image_path[t]) for t in (ta, tb)])
                masks = np.stack([tifffile.imread(fr.mask_path[t]) for t in (ta, tb)])
                pred = model._predict(imgs, masks)
                node = {n["id"]: (int(n["time"]), int(n["label"])) for n in pred["nodes"]}
                best = {}  # target label -> (weight, anchor label)
                for (u, v), wt in pred["weights"]:
                    (t0, l0), (t1, l1) = sorted([node[int(u)], node[int(v)]])
                    if t0 == 0 and t1 == 1 and wt > best.get(l1, (-1, 0))[0]:
                        best[l1] = (float(wt), l0)
                graph, _ = model.track(imgs, masks, mode="greedy", progbar_class=lambda x, **k: x)
                greedy = {}
                for u, v in graph.edges:
                    du, dv = graph.nodes[u], graph.nodes[v]
                    if int(du["time"]) == 0 and int(dv["time"]) == 1:
                        greedy[int(dv["label"])] = int(du["label"])

                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                anc, tgt = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
                ro = {int(v): k for k, v in enumerate(tgt.local_detection_id)}
                co = {int(v): k for k, v in enumerate(anc.local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok):
                    continue
                r = ok.target_detection_id.map(ro).to_numpy()
                truth = ok.ancestor_detection_id.map(co).to_numpy()
                by_track = {int(t): k for k, t in enumerate(anc.gt_track_id)}
                near = cKDTree(anc[["center_y", "center_x"]].to_numpy()).query(
                    tgt[["center_y", "center_x"]].to_numpy())[1]
                labels = tgt.gt_track_id.astype(int).to_numpy()
                for name, link in (("trackastra_greedy", greedy),
                                   ("trackastra_argmax", {k: v[1] for k, v in best.items()})):
                    g = np.array([by_track.get(link.get(lb, -1), near[k]) for k, lb in enumerate(labels)])
                    g = g[r]
                    rows.append({"sequence": alias, "H": h, "window_id": w["window_id"], "n_anchors": len(co),
                                 "n_scored": len(r), "method": name, "accuracy": float(np.mean(g == truth)),
                                 "set_f1": family_f1(truth, g),
                                 "linked": float(np.mean([lb in link for lb in labels[r]]))})
                g = near[r]
                rows.append({"sequence": alias, "H": h, "window_id": w["window_id"], "n_anchors": len(co),
                             "n_scored": len(r), "method": "nearest", "accuracy": float(np.mean(g == truth)),
                             "set_f1": family_f1(truth, g), "linked": 1.0})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"{a.out}.parquet", index=False)
    df["dense"] = df.n_anchors > 200
    print(df.pivot_table(index="method", columns="H", values="accuracy").round(4).to_string())
    print("dense\n", df[df.dense].pivot_table(index="method", columns="H", values="accuracy").round(4).to_string())
    print("set_f1\n", df.pivot_table(index="method", columns="H", values="set_f1").round(4).to_string())
    print("linked\n", df.pivot_table(index="method", columns="H", values="linked").round(4).to_string())


if __name__ == "__main__":
    main()
