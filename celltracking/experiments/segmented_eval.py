"""End-to-end ancestry from a real segmenter's detections (instead of GT positions).

Detections (segment_frames.py) are matched to GT cells through the GT SEG label
under each detection's centroid (one detection per GT cell, closest wins).
Two scores per window:
  matched     accuracy over valid GT targets whose target and true ancestor were both detected
  end_to_end  the same correct count divided by all valid GT targets (misses count as errors)

  python experiments/segmented_eval.py --seg outputs/seg/cpsam --tags raft_v3 raft_sim_loc
  python experiments/segmented_eval.py --seg gt --out segmented_eval_gt   (GT cells as detections)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import expansion_compensate  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, SEQUENCE  # noqa: E402
from experiments.lineage_glue import INDEX, frame_dict, image_shape_of, mean_field  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402


def match(det, gt, seg):
    """Map GT track id -> detection row (closest detection whose centroid lies in that GT mask)."""
    yy = np.clip(np.round(det.center_y.to_numpy()).astype(int), 0, seg.shape[0] - 1)
    xx = np.clip(np.round(det.center_x.to_numpy()).astype(int), 0, seg.shape[1] - 1)
    lab = seg[yy, xx]
    centre = dict(zip(gt.gt_track_id.astype(int), gt[["center_y", "center_x"]].to_numpy()))
    best = {}
    for k, t in enumerate(lab):
        t = int(t)
        if t == 0 or t not in centre:
            continue
        d = np.hypot(*(det[["center_y", "center_x"]].to_numpy()[k] - centre[t]))
        if t not in best or d < best[t][1]:
            best[t] = (k, d)
    return {t: k for t, (k, _) in best.items()}


def main():
    from cellanc.loader import Windows
    from experiments.family_set import examples
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=str(ROOT / "outputs/seg/cpsam"), help="detection dir, or 'gt'")
    ap.add_argument("--tags", nargs="+", default=["raft_v3", "raft_sim_loc"])
    ap.add_argument("--out", default="segmented_eval_cpsam")
    a = ap.parse_args()
    seg_dir = Path(a.seg)
    models = {t: load(t) for t in a.tags}
    windows = Windows(BENCH, ROOT / "data").windows
    rows, det_rows = [], []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        masks = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index").mask_path
        cache = {}

        def frame(t, gt):
            if t not in cache:
                if a.seg == "gt":  # true positions through the same code path
                    det = gt.reset_index(drop=True)
                    m = {int(g): k for k, g in enumerate(det.gt_track_id)}
                else:
                    det = pd.read_parquet(seg_dir / alias / f"{t:05d}.parquet").reset_index(drop=True)
                    m = match(det, gt, tifffile.imread(masks[t]))
                cache[t] = (det, m)
                det_rows.append({"sequence": alias, "frame": t, "n_gt": len(gt), "n_det": len(det),
                                 "recall": len(m) / max(len(gt), 1), "precision": len(m) / max(len(det), 1)})
            return cache[t]

        for h in (33, 66, 132):
            for e in examples(alias, windows, compensate=False, horizon=h):
                w = e["window"]
                anc_gt, tgt_gt = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
                da, ma = frame(w["anchor_frame"], anc_gt)
                db, mb = frame(w["target_frame"], tgt_gt)
                if len(da) < 2 or len(db) < 1:
                    continue
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                gid_t = dict(zip(tgt_gt.local_detection_id.astype(int), tgt_gt.gt_track_id.astype(int)))
                gid_a = dict(zip(anc_gt.local_detection_id.astype(int), anc_gt.gt_track_id.astype(int)))
                ok = ok[ok.target_detection_id.isin(gid_t) & ok.ancestor_detection_id.isin(gid_a)]
                if not len(ok):
                    continue
                r, truth = [], []
                for td, ad in zip(ok.target_detection_id.astype(int), ok.ancestor_detection_id.astype(int)):
                    kt, ka = mb.get(gid_t[td]), ma.get(gid_a[ad])
                    if kt is not None and ka is not None:
                        r.append(kt)
                        truth.append(ka)
                n_valid = len(ok)
                if not r:
                    continue
                r, truth = np.array(r), np.array(truth)
                pa, pb = da[["center_y", "center_x"]].to_numpy(float), db[["center_y", "center_x"]].to_numpy(float)
                preds = {"nearest": cKDTree(pa).query(pb)[1],
                         "nearest_comp": cKDTree(pa).query(expansion_compensate(
                             np.column_stack([np.arange(len(pa)), pa]), np.column_stack([np.arange(len(pb)), pb]))[:, 1:])[1]}
                for tag, (model, margs) in models.items():
                    prob, fam, _ = predict(model, frame_dict(da), frame_dict(db), h, shape,
                                           margs.get("no_shape", False), init=margs.get("init", "radial"))
                    q = prob[:, :-1] / np.maximum(prob[:, :-1].sum(1, keepdims=True), 1e-9)
                    preds[f"{tag}_argmax"] = q.argmax(1)
                    i, j, wf = fam
                    preds[f"{tag}_mf_fam2"] = mean_field(q, i, j, wf, 2).argmax(1) if len(i) else q.argmax(1)
                for name, p in preds.items():
                    correct = int(np.sum(p[r] == truth))
                    rows.append({"sequence": alias, "H": h, "window_id": w["window_id"], "n_anchors": len(anc_gt),
                                 "method": name, "n_valid": n_valid, "n_matched": len(r),
                                 "matched": correct / len(r), "end_to_end": correct / n_valid})
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"{a.out}.parquet", index=False)
    det = pd.DataFrame(det_rows)
    det.to_parquet(RESULTS / f"{a.out}_detection.parquet", index=False)
    print(det[["recall", "precision"]].describe().round(3).to_string())
    for col in ("matched", "end_to_end"):
        print(col, "\n", df.pivot_table(index="method", columns="H", values=col).round(3).to_string())
        print("dense\n", df[df.n_anchors > 200].pivot_table(index="method", columns="H", values=col).round(3).to_string())


if __name__ == "__main__":
    main()
