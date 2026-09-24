"""CLI for the cell-ancestry data pipeline (docs/260924_cell_ancestry_data_guide.md).

  python main.py fetch  sim_plus one_in_a_million   # download + md5 + manifest
  python main.py index  sim_plus                     # tables -> data/index, audit -> data/audit
  python main.py windows sim_plus 01 --a 10 --b 60 --strides 1 2 4 8
"""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from cellanc.audit import audit, build_index
from cellanc.ctc import find_sequences
from cellanc.windows import make_window

DATA = Path(__file__).parent / "data"

# name -> (url, zip name, published md5 or None, time step in minutes) — doc §2
SOURCES = {
    "one_in_a_million": ("https://zenodo.org/records/7260137/files/ctc_format.zip?download=1",
                         "ctc_format.zip", "f29fddadcee5c18c86716d3958c5e0da", 1.0),
    "sim_plus": ("https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip",
                 "Fluo-N2DH-SIM+.zip", None, 29.0),
    "hsc": ("https://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip",
            "BF-C2DL-HSC.zip", None, 5.0),
    "musc": ("https://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip",
             "BF-C2DL-MuSC.zip", None, 5.0),
    "hela": ("https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip",
             "Fluo-N2DL-HeLa.zip", None, 30.0),
}


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
        url, fname, want, _ = SOURCES[name]
        zp = DATA / "downloads" / fname
        subprocess.run(["curl", "-fL", "-#", "--retry", "3", "-C", "-", url, "-o", str(zp)], check=True)
        got = md5(zp)
        if want and got != want:
            raise SystemExit(f"{name}: md5 {got} != published {want}")
        with ZipFile(zp) as z:
            if bad := z.testzip():
                raise SystemExit(f"{name}: corrupt member {bad}")
            infos = z.infolist()
            size = sum(i.file_size for i in infos) / 2**30
            print(f"{name}: {sum(not i.is_dir() for i in infos)} files, {size:.2f} GiB uncompressed")
            if size > extract_limit_gib:
                raise SystemExit(f"{name}: {size:.1f} GiB exceeds --extract-limit-gib")
            z.extractall(DATA / "raw" / name)
        manifest[name] = {"url": url, "file": str(zp.relative_to(DATA)), "md5": got,
                          "md5_published": want, "uncompressed_gib": round(size, 3),
                          "downloaded": dt.date.today().isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2))


def index(name: str, workers: int):
    rows = []
    for seq in find_sequences(name, DATA / "raw" / name):
        print(f"indexing {name}/{seq.seq} ...", flush=True)
        frames, tracks, obs = build_index(seq, workers)
        summary, issues = audit(frames, tracks, obs)
        summary["time_step_min"] = SOURCES[name][3]
        out = DATA / "index" / name / seq.seq
        out.mkdir(parents=True, exist_ok=True)
        frames.assign(shape=frames["shape"].astype(str),
                      marker_shape=frames["marker_shape"].astype(str)).to_parquet(out / "frames.parquet")
        tracks.to_parquet(out / "tracks.parquet")
        obs.to_parquet(out / "observations.parquet")  # evaluator-side: contains gt_track_id
        aud = DATA / "audit" / name
        aud.mkdir(parents=True, exist_ok=True)
        issues.to_csv(aud / f"{seq.seq}_issues.csv", index=False)
        rows.append(summary)
        print(json.dumps(summary, default=str))
    pd.DataFrame(rows).to_csv(DATA / "audit" / name / "summary.csv", index=False)


def windows(name: str, seq_id: str, a: int, b: int, strides, seed: int):
    src = DATA / "index" / name / seq_id
    obs = pd.read_parquet(src / "observations.parquet")
    T = {r.gt_track_id: (r.start_frame, r.end_frame, r.parent_id)
         for r in pd.read_parquet(src / "tracks.parquet").itertuples()}
    root = DATA / "benchmarks" / "v0"
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for s in strides:
        stride = None if s == 0 else s
        inp, mapping, targets, sibs = make_window(obs, T, a, b, stride, SOURCES[name][3], seed,
                                                  f"{name}_{seq_id}")
        wid = inp["window_id"]
        (root / "inputs" / f"{wid}.json").write_text(json.dumps(inp))
        pd.DataFrame(mapping).to_csv(root / "labels" / f"{wid}_gt_mapping.csv", index=False)
        t = pd.DataFrame(targets).astype({"ancestor_detection_id": "Int64",
                                          "hidden_intermediates": "Int64"})
        t.to_csv(root / "labels" / f"{wid}_targets.csv", index=False)
        pd.DataFrame(sibs, columns=["window_id", "det_1", "det_2"]).to_csv(
            root / "labels" / f"{wid}_siblings.csv", index=False)
        print(wid, "frames", inp["observed_frames"][:6], "... |",
              t.target_status.value_counts().to_dict(), "| depth",
              t.generation_depth.value_counts().sort_index().to_dict(), "| sibling pairs", len(sibs))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("names", nargs="+", choices=SOURCES)
    f.add_argument("--extract-limit-gib", type=float, default=30)
    i = sub.add_parser("index")
    i.add_argument("name", choices=SOURCES)
    i.add_argument("--workers", type=int, default=8)
    w = sub.add_parser("windows")
    w.add_argument("name", choices=SOURCES)
    w.add_argument("seq")
    w.add_argument("--a", type=int, required=True)
    w.add_argument("--b", type=int, required=True)
    w.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8, 0], help="0 = endpoints")
    w.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.names, args.extract_limit_gib)
    elif args.cmd == "index":
        index(args.name, args.workers)
    else:
        windows(args.name, args.seq, args.a, args.b, args.strides, args.seed)


if __name__ == "__main__":
    main()
