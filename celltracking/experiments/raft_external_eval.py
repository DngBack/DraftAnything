"""Evaluate LineageRAFT checkpoints on OIAM and every external dataset.

Same window loops, detection shuffles (seed 20261002) and metrics as
radial_sector_probe.py / rpsm_radial_external_test.py. All external sets have
been viewed before; for an OIAM-trained model these are development numbers.
For a simulation-only model no real lineage label entered training or model
selection, which is the point of the comparison.

  python experiments/raft_external_eval.py --tags raft_v1_loc raft_sim_loc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.fast_microcolony_external_test import (  # noqa: E402
    GAPS as FAST_GAPS, HELD_OUT as FAST_HELD_OUT, family_f1,
    load_colony as load_fast, make_window as fast_window)
from experiments.fetch_rpsm import HELD_OUT as RPSM_HELD_OUT  # noqa: E402
from experiments.lfct_example_external_test import XML, load_graph, make_window as lfct_window  # noqa: E402
from experiments.lineage_glue import image_shape_of  # noqa: E402
from experiments.lineage_raft import decoders, load, predict  # noqa: E402
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402
from experiments.reca_microcolony_external_test import (  # noqa: E402
    GAPS as RECA_GAPS, HELD_OUT as RECA_HELD_OUT, make_window as reca_window)
from experiments.reca_schnitzcells import load_colony as load_schnitz  # noqa: E402
from experiments.tui_t98g_external_test import (  # noqa: E402
    GAPS as T98G_GAPS, MASK_DIR, frame_table, lineage, target_labels)


def plain(df):
    n = len(df)
    return {"pos": df[["center_y", "center_x"]].to_numpy(np.float64),
            "axis": np.zeros((n, 2)), "major": np.ones(n), "minor": np.ones(n)}


def bbox_shape(frames_xy):
    hi = np.max([f.max(0) for f in frames_xy if len(f)], axis=0)
    return tuple((hi * 1.05 + 5).tolist())


class Runner:
    def __init__(self, tags, decoder_names, ensemble=False):
        self.models = {t: load(t) for t in tags}
        self.ensemble = ensemble
        self.decoder_names = decoder_names
        self.rows = []

    def window(self, dataset, seq, wid, minutes, e, valid, truth, shape):
        if not len(truth):
            return
        preds = dict(geometry_assignments(e))
        fa, fb = plain(e["anchor"]), plain(e["target"])
        probs = []
        for tag, (model, margs) in self.models.items():
            if len(fa["pos"]) < 2:
                for d in self.decoder_names:
                    preds[f"{tag}_{d}"] = np.zeros(len(fb["pos"]), int)
                continue
            prob, fam, _ = predict(model, fa, fb, minutes, shape, no_shape=True,
                                   init=margs.get("init", "radial"))
            p = prob[:, :-1]
            probs.append(p / np.maximum(p.sum(1, keepdims=True), 1e-9))
            if self.decoder_names == ["argmax"]:  # skip the slow mean-field variants
                dec = {"argmax": (p / np.maximum(p.sum(1, keepdims=True), 1e-9)).argmax(1)}
            else:
                dec = decoders(prob, fam)
            for d in self.decoder_names:
                preds[f"{tag}_{d}"] = dec[d]
        if self.ensemble:  # mean of the tags' normalised probabilities
            preds["ens_argmax"] = np.mean(probs, 0).argmax(1) if probs else np.zeros(len(fb["pos"]), int)
        for method, p in preds.items():
            g = p[valid]
            self.rows.append({"dataset": dataset, "sequence": seq, "window_id": wid,
                              "gap_minutes": minutes, "n_anchors": len(fa["pos"]),
                              "n_targets": len(fb["pos"]), "n_scored": len(truth),
                              "method": method, "accuracy": float(np.mean(g == truth)),
                              "set_f1": family_f1(truth, g)})


def run(args):
    r = Runner(args.tags, args.decoders, args.ensemble)
    rng = np.random.default_rng(20261002)
    sets = set(args.datasets)
    if "OIAM" in sets:
        windows = Windows(BENCH, ROOT / "data").windows
        for alias in ("M3", "M4"):
            shape = image_shape_of(alias)
            for h in (33, 66, 132):
                for e in examples(alias, windows, compensate=False, horizon=h):
                    tg = e["targets"]
                    ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                    ro = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
                    co = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
                    ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                    r.window("OIAM", alias, e["window"]["window_id"], h, e,
                             ok.target_detection_id.map(ro).to_numpy(),
                             ok.ancestor_detection_id.map(co).to_numpy(), shape)
            print("OIAM", alias, flush=True)
    for name, held, base, gaps, mpf in (("recA", RECA_HELD_OUT, "reca_holdout", RECA_GAPS, 7.5),
                                       ("rpsM", RPSM_HELD_OUT, "rpsm_holdout", (10, 20, 40), 3)):
        if name not in sets:
            continue
        for colony in held:
            frames, mapping, tracks, invalid, _ = load_schnitz(ROOT / "data/raw" / base / colony)
            shape = bbox_shape([f[:, ::-1] for f in frames.values() if len(f)])
            for gap in gaps:
                for start in sorted(frames):
                    if start + gap not in frames or not len(frames[start]) or not len(frames[start + gap]):
                        continue
                    e, valid, truth = reca_window(start, gap, frames, mapping, tracks, invalid, rng)
                    r.window(name, colony, f"{start}_{start+gap}", gap * mpf, e, valid, truth, shape)
            print(name, colony, flush=True)
    if "FAST" in sets:
        for colony in FAST_HELD_OUT:
            frames, mapping, tracks, _ = load_fast(ROOT / "data/downloads" / f"fast_{colony}.zip")
            shape = bbox_shape([f[:, ::-1] for f in frames.values() if len(f)])
            for gap in FAST_GAPS:
                for start in sorted(frames):
                    if start + gap not in frames:
                        continue
                    e, valid, truth = fast_window(start, gap, frames, mapping, tracks, rng)
                    r.window("FAST", colony, f"{start}_{start+gap}", gap * 5, e, valid, truth, shape)
            print("FAST", colony, flush=True)
    if "LFCT" in sets:
        spots, by_frame, parent = load_graph(XML)
        allxy = np.array([[s[1], s[2]] for s in spots.values()])
        shape = bbox_shape([allxy])
        for gap in (2, 4, 8):
            for start in sorted(by_frame):
                if start + gap not in by_frame:
                    continue
                e, valid, truth = lfct_window(start, gap, spots, by_frame, parent, rng)
                r.window("LFCT", "FLD_3", f"{start}_{start+gap}", gap * 15, e, valid, truth, shape)
        print("LFCT", flush=True)
    if "T98G" in sets:
        parent = lineage(MASK_DIR / "man_track.txt")
        frames = {t: frame_table(MASK_DIR / f"mask{t:03}.tif", t) for t in range(37)}
        shape = bbox_shape([f[0][["center_y", "center_x"]].to_numpy() for f in frames.values()])
        for _, gap in T98G_GAPS.items():
            for start in range(37 - gap):
                anchor, source_ids = frames[start]
                target, target_ids = frames[start + gap]
                labels, _ = target_labels(source_ids, target_ids, parent)
                vl = labels[labels.target_status == "valid_anchor"]
                valid = vl.target_detection_id.to_numpy(dtype=int) - 1
                truth = vl.ancestor_detection_id.to_numpy(dtype=int) - 1
                a = anchor[["center_y", "center_x"]].to_numpy()
                b = target[["center_y", "center_x"]].to_numpy()
                e = {"anchor": anchor, "target": target, "window": {"horizon_frames": gap * 10},
                     "cost": np.sum((b[:, None] - a[None]) ** 2, -1).astype(np.float32)}
                r.window("T98G", "T98G", f"{start}_{start+gap}", gap * 10, e, valid, truth, shape)
        print("T98G", flush=True)
    out = pd.DataFrame(r.rows)
    out.to_parquet(RESULTS / f"raft_external_{args.out}.parquet", index=False)
    summary(out)


def summary(out):
    seq = out.groupby(["dataset", "gap_minutes", "sequence", "method"]).accuracy.mean()
    print(seq.groupby(["dataset", "gap_minutes", "method"]).mean().unstack("method").round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--decoders", nargs="+", default=["argmax", "mf_fam2"])
    p.add_argument("--datasets", nargs="+", default=["OIAM", "recA", "rpsM", "FAST", "LFCT", "T98G"])
    p.add_argument("--out", default="v1")
    p.add_argument("--ensemble", action="store_true", help="also score the mean of the tags' probabilities")
    run(p.parse_args())


if __name__ == "__main__":
    main()
