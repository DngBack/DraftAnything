#!/usr/bin/env python3
"""Extract per-tile DINOv2-S/14 embeddings for every image of the LCO manifest (train+val+test).

    python scripts/extract.py --kinds cls patchmean --frame_batch_size 16 --num_workers 8

Output: cache/features_<kind>.npy (N, 16, 384) fp16 + .json sidecar. About 2.4 GB per kind.
Extraction is unsupervised and identical for every split, so features of held-out images carry no
label information; everything fitted on features (PCA, bandwidth, prototypes) is fitted later on
training images only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from popev.features import extract  # noqa: E402
from popev.phoebi import PHOEBI_ROOT, Manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=PHOEBI_ROOT / "data")
    ap.add_argument("--kinds", nargs="+", default=["cls"])
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--frame_batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images only (tag _smoke)")
    a = ap.parse_args()
    man = Manifest(a.data_dir / "splits_lco.json")
    paths = man.paths[: a.limit] if a.limit else man.paths
    extract(paths, a.kinds, a.backbone, a.frame_batch_size, a.num_workers, a.device, tag="_smoke" if a.limit else "")


if __name__ == "__main__":
    main()
