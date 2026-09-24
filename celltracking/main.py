"""CLI for the cell-ancestry data pipeline (docs/260924_cell_ancestry_data_guide.md).

Inputs (downloaded, immutable) live in data/; everything the pipeline produces goes to outputs/.

  python main.py fetch one_in_a_million sim_plus   # data/downloads, data/raw, data/manifest.json
  python main.py index one_in_a_million            # outputs/index, outputs/audit
  python main.py overlays one_in_a_million 00      # outputs/audit/overlays
  python main.py build                             # outputs/splits, outputs/benchmarks/v0
  python main.py baseline                          # outputs/results
  python main.py stats                             # outputs/tables, outputs/figures (expA/expB)
  python main.py figures                           # outputs/figures/fig*.png for the report
"""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from cellanc.audit import audit, build_index, classify_track_ends
from cellanc.baselines import BASELINES, mutual_nearest_siblings
from cellanc.benchmark import SeqData, build_sequence, cell_cycle_frames, write_sequence
from cellanc.ctc import find_sequences
from cellanc.evaluate import score_window
from cellanc.loader import Windows
from cellanc.visual import division_sheet, end_sheet

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
BENCH = OUT / "benchmarks" / "v0"
SEED = 0

CTC = "https://data.celltrackingchallenge.net/training-datasets/"
# Source facts (doc §2, §6; Zenodo 7260137; celltrackingchallenge.net/2d-datasets) plus our
# design choices: aliases, splits, border_px used only to classify track ends.
DATASETS = {
    "one_in_a_million": dict(
        url="https://zenodo.org/records/7260137/files/ctc_format.zip?download=1",
        zip="ctc_format.zip", md5="f29fddadcee5c18c86716d3958c5e0da", time_step_min=1.0,
        pixel_size_um=0.072, modality="transmitted light (DIA), C. glutamicum, microfluidic",
        experiment_id="oiam_single_experiment_5_chambers", foi_margin_px=0, border_px=30,
        annotation_type="manually corrected SEG+TRA, all frames",
        aliases={"00": "M0", "01": "M1", "02": "M2", "03": "M3", "04": "M4"},
        splits={"M0": "train", "M1": "train", "M2": "train", "M3": "validation", "M4": "test"}),
    "sim_plus": dict(
        url=CTC + "Fluo-N2DH-SIM+.zip", zip="Fluo-N2DH-SIM+.zip", md5=None, time_step_min=29.0,
        pixel_size_um=0.125, modality="simulated fluorescence nuclei (HL60)",
        experiment_id="simulation", foi_margin_px=0, border_px=10,
        annotation_type="CTC GT (simulated)",
        aliases={"01": "SIM-01", "02": "SIM-02"}, splits={"SIM-01": "sanity", "SIM-02": "sanity"}),
    "hsc": dict(
        url=CTC + "BF-C2DL-HSC.zip", zip="BF-C2DL-HSC.zip", md5=None, time_step_min=5.0,
        pixel_size_um=0.645, modality="brightfield, mouse HSC", experiment_id=None,
        foi_margin_px=25, border_px=40, annotation_type="CTC GT TRA + sparse gold SEG",
        aliases={"01": "HSC-01", "02": "HSC-02"},
        splits={"HSC-01": "domain_adapt", "HSC-02": "domain_test"}),
    "musc": dict(
        url=CTC + "BF-C2DL-MuSC.zip", zip="BF-C2DL-MuSC.zip", md5=None, time_step_min=5.0,
        pixel_size_um=0.645, modality="brightfield, mouse MuSC", experiment_id=None,
        foi_margin_px=25, border_px=40, annotation_type="CTC GT TRA + sparse gold SEG",
        aliases={"01": "MuSC-01", "02": "MuSC-02"},
        splits={"MuSC-01": "domain_adapt", "MuSC-02": "domain_test"}),
    "hela": dict(
        url=CTC + "Fluo-N2DL-HeLa.zip", zip="Fluo-N2DL-HeLa.zip", md5=None, time_step_min=30.0,
        pixel_size_um=0.645, modality="fluorescence nuclei H2b-GFP, HeLa", experiment_id=None,
        foi_margin_px=25, border_px=40, annotation_type="CTC GT TRA + sparse gold SEG",
        aliases={"01": "HeLa-01", "02": "HeLa-02"},
        splits={"HeLa-01": "domain_adapt", "HeLa-02": "domain_test"}),
}
# Splits whose labels may be used to choose horizons (never validation/test/domain_test).
HORIZON_SPLITS = {"train", "sanity", "domain_adapt"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(names, extract_limit_gib: float):
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    (DATA / "downloads").mkdir(parents=True, exist_ok=True)
    for name in names:
        cfg = DATASETS[name]
        zp = DATA / "downloads" / cfg["zip"]
        subprocess.run(["curl", "-fL", "-#", "--retry", "3", "-C", "-", cfg["url"], "-o", str(zp)],
                       check=True)
        got = md5(zp)
        if cfg["md5"] and got != cfg["md5"]:
            raise SystemExit(f"{name}: md5 {got} != published {cfg['md5']}")
        with ZipFile(zp) as z:
            if bad := z.testzip():
                raise SystemExit(f"{name}: corrupt member {bad}")
            infos = z.infolist()
            size = sum(i.file_size for i in infos) / 2**30
            print(f"{name}: {sum(not i.is_dir() for i in infos)} files, {size:.2f} GiB uncompressed")
            if size > extract_limit_gib:
                raise SystemExit(f"{name}: {size:.1f} GiB exceeds --extract-limit-gib")
            z.extractall(DATA / "raw" / name)
        manifest[name] = {"url": cfg["url"], "file": str(zp.relative_to(DATA)), "md5": got,
                          "md5_published": cfg["md5"], "zip_bytes": zp.stat().st_size,
                          "uncompressed_gib": round(size, 3),
                          "downloaded": dt.date.today().isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2))


def index(name: str, workers: int):
    cfg = DATASETS[name]
    rows = []
    for seq in find_sequences(name, DATA / "raw" / name):
        print(f"indexing {name}/{seq.seq} ...", flush=True)
        frames, tracks, obs, seg = build_index(seq, workers)
        summary, issues = audit(frames, tracks, obs)
        shape = frames["shape"].dropna().iat[0]
        tracks = classify_track_ends(tracks, obs, shape, int(frames.frame_index.max()),
                                     cfg["border_px"])
        summary.update({f"end_{k}": v for k, v in tracks.end_reason.value_counts().items()})
        summary.update({f"start_{k}": v for k, v in tracks.start_reason.value_counts().items()})
        if len(seg):
            summary.update({f"seg_{k}": int(seg[k].sum()) for k in
                            ["n_seg", "n_tra", "one_to_one", "tra_unmatched", "seg_unmatched",
                             "ambiguous"]})
        out = OUT / "index" / name / seq.seq
        out.mkdir(parents=True, exist_ok=True)
        frames.assign(shape=frames["shape"].astype(str),
                      marker_shape=frames["marker_shape"].astype(str)).to_parquet(out / "frames.parquet")
        tracks.to_parquet(out / "tracks.parquet")
        obs.to_parquet(out / "observations.parquet")  # evaluator-side: contains gt_track_id
        aud = OUT / "audit" / name
        aud.mkdir(parents=True, exist_ok=True)
        issues.to_csv(aud / f"{seq.seq}_issues.csv", index=False)
        seg.to_csv(aud / f"{seq.seq}_seg_tra_match.csv", index=False)
        rows.append(summary)
        print(json.dumps(summary, default=str))
    pd.DataFrame(rows).to_csv(OUT / "audit" / name / "summary.csv", index=False)


def overlays(name: str, seq_id: str):
    seq = next(s for s in find_sequences(name, DATA / "raw" / name) if s.seq == seq_id)
    src = OUT / "index" / name / seq_id
    tracks, obs = pd.read_parquet(src / "tracks.parquet"), pd.read_parquet(src / "observations.parquet")
    out = OUT / "audit" / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    r = {"one_in_a_million": 40, "sim_plus": 40}.get(name, 30)
    division_sheet(seq, tracks, obs, out / f"{name}_{seq_id}_divisions.png", r=r, seed=SEED)
    end_sheet(seq, tracks, obs, out / f"{name}_{seq_id}_ends.png", r=r, seed=SEED)
    print("wrote", out)


def _load_seq(name, seq_id):
    cfg = DATASETS[name]
    src = OUT / "index" / name / seq_id
    frames = pd.read_parquet(src / "frames.parquet")
    tr = pd.read_parquet(src / "tracks.parquet")
    alias = cfg["aliases"][seq_id]
    shape = tuple(int(v) for v in frames["shape"].iat[0].strip("()").split(","))
    return SeqData(name, seq_id, alias, cfg["splits"][alias], cfg["time_step_min"],
                   cfg["foi_margin_px"], shape, pd.read_parquet(src / "observations.parquet"),
                   {r.gt_track_id: (r.start_frame, r.end_frame, r.parent_id) for r in tr.itertuples()},
                   frames)


def build(names, min_anchors: int):
    manifest = json.loads((DATA / "manifest.json").read_text())
    seq_rows, splits, horizons_cfg = [], {}, {}
    for name in names:
        cfg = DATASETS[name]
        seqs = [_load_seq(name, s) for s in sorted(cfg["aliases"])
                if (OUT / "index" / name / s).exists()]
        # Horizons from the cell-cycle distribution of label-usable splits only (doc §10).
        cyc = np.concatenate([cell_cycle_frames(s.tracks) for s in seqs if s.split in HORIZON_SPLITS])
        C = int(round(np.median(cyc)))
        horizons = sorted({max(2, C // 2), C, 2 * C})
        deltas = [d for d in (2**k for k in range(12)) if d <= 4 * C]
        a_step = max(1, C // 2)
        horizons_cfg[name] = {"cell_cycle_median_frames": C, "cell_cycle_iqr_frames":
                              np.percentile(cyc, [25, 75]).tolist(), "n_cycles": len(cyc),
                              "horizons_frames": horizons, "deltas_frames": deltas,
                              "a_step_frames": a_step, "min_anchors": min_anchors,
                              "cycle_source_splits": sorted({s.split for s in seqs} & HORIZON_SPLITS)}
        for s in seqs:
            print(f"building {s.alias} ({s.split}) ...", flush=True)
            local, windows, targets, sibs = build_sequence(s, horizons, a_step, min_anchors,
                                                           deltas, SEED)
            write_sequence(BENCH, DATA, s, local, windows, targets, sibs)
            splits.setdefault(name, {}).setdefault(s.split, []).append(s.alias)
            seq_rows.append({
                "dataset": name, "sequence_id": s.sequence_id, "alias": s.alias, "split": s.split,
                "experiment_id": cfg["experiment_id"], "modality": cfg["modality"],
                "pixel_size_um": cfg["pixel_size_um"], "time_step_min": cfg["time_step_min"],
                "foi_margin_px": cfg["foi_margin_px"], "source_url": cfg["url"],
                "checksum_md5": manifest[name]["md5"], "annotation_type": cfg["annotation_type"],
                "n_windows": len(windows), "n_target_rows": len(targets)})
            print(f"  {len(windows)} windows, {len(targets)} target rows", flush=True)
    (OUT / "index").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seq_rows).to_csv(OUT / "index" / "sequences.csv", index=False)
    (OUT / "splits").mkdir(exist_ok=True)
    (OUT / "splits" / "v0.json").write_text(json.dumps(
        {"note": "custom split on public training data; not an official CTC/source split",
         "unit": "movie", "seed": SEED, "splits": splits}, indent=2))
    (BENCH / "config.json").write_text(json.dumps(horizons_cfg, indent=2))


def _run_seq(alias):
    ws = Windows(BENCH, DATA)
    ws.windows = [w for w in ws.windows if w["sequence_id"] == alias]
    tg = pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
    sb = pd.read_parquet(BENCH / "labels/siblings" / f"{alias}.parquet")
    tg_by, sb_by = dict(list(tg.groupby("window_id"))), dict(list(sb.groupby("window_id")))
    empty_sb = sb.iloc[:0]
    rows = []
    for w in ws.windows:
        win = ws.load(w)
        sibs = mutual_nearest_siblings(win)
        for bname, fn in BASELINES.items():
            s = score_window(fn(win), sibs, tg_by[w["window_id"]], sb_by.get(w["window_id"], empty_sb))
            rows.append({"baseline": bname, **{k: w[k] for k in
                         ["window_id", "split", "dataset", "sequence_id", "anchor_frame",
                          "target_frame", "horizon_frames", "horizon_min", "schedule_id"]},
                         "experiments": ",".join(w["experiments"]),
                         "n_observed": len(w["observed_frames"]),
                         "n_anchors": len(win["detections"][w["anchor_frame"]]), **s})
    return pd.DataFrame(rows)


def baseline(workers: int):
    aliases = sorted(p.stem for p in (BENCH / "inputs/windows").glob("*.jsonl"))
    with ProcessPoolExecutor(workers) as ex:
        res = pd.concat(ex.map(_run_seq, aliases))
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / "results" / "baselines_v0.parquet", index=False)
    print(len(res), "window-baseline rows")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("names", nargs="+", choices=DATASETS)
    f.add_argument("--extract-limit-gib", type=float, default=30)
    i = sub.add_parser("index")
    i.add_argument("name", choices=DATASETS)
    i.add_argument("--workers", type=int, default=16)
    o = sub.add_parser("overlays")
    o.add_argument("name", choices=DATASETS)
    o.add_argument("seq")
    b = sub.add_parser("build")
    b.add_argument("names", nargs="*", default=list(DATASETS))
    b.add_argument("--min-anchors", type=int, default=8)
    r = sub.add_parser("baseline")
    r.add_argument("--workers", type=int, default=16)
    sub.add_parser("stats")
    g = sub.add_parser("figures")
    g.add_argument("only", nargs="*")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.names, args.extract_limit_gib)
    elif args.cmd == "index":
        index(args.name, args.workers)
    elif args.cmd == "overlays":
        overlays(args.name, args.seq)
    elif args.cmd == "build":
        build([n for n in args.names if (OUT / "index" / n).exists()], args.min_anchors)
    elif args.cmd == "baseline":
        baseline(args.workers)
    elif args.cmd == "stats":
        from cellanc.stats import tables
        tables(OUT)
    else:
        from cellanc.figures import make_all
        make_all(OUT, DATA, DATASETS, args.only or None)


if __name__ == "__main__":
    main()
