"""Per-image tile embeddings in one fp16 memmap: (N_images, T, D), rows in Manifest order.

Extraction reuses PHOEBI's `UnitExtractor`, so illumination correction (divide, sigma 64), the
4x4 grid of 224 px tiles and the frozen DINOv2-S/14 read-out are the paper's. Kind "cls" is the
paper's unit; "patchmean" is the mean patch token of the same tile. Units are L2-normalised by
PHOEBI's extractor. A JSON sidecar records everything the cache depends on; `open_store` refuses a
cache whose sidecar does not match the manifest.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

from popev.phoebi import PHOEBI_ROOT

CACHE = Path(__file__).resolve().parents[1] / "cache"


def paths_digest(paths: List[str]) -> str:
    return hashlib.sha1("\n".join(paths).encode()).hexdigest()


def store_path(kind: str, tag: str = "") -> Path:
    return CACHE / f"features_{kind}{tag}.npy"


def open_store(paths: List[str], kind: str = "cls", tag: str = "") -> np.memmap:
    p = store_path(kind, tag)
    meta = json.loads(p.with_suffix(".json").read_text())
    if meta["paths_sha1"] != paths_digest(paths) or not meta.get("complete"):
        raise SystemExit(f"{p}: cache does not match the manifest or is incomplete; re-run scripts/extract.py")
    return np.load(p, mmap_mode="r")


def extract(paths: List[str], kinds: List[str], backbone: str, frame_batch_size: int, num_workers: int,
            device: str = "cuda", tag: str = "") -> None:
    import torch
    if str(PHOEBI_ROOT) not in sys.path:
        sys.path.insert(0, str(PHOEBI_ROOT))
    from src.compositional.units import UnitExtractor, iter_image_units

    dev = torch.device(device)
    ex = UnitExtractor(backbone, tile_size=224, grid=4, kinds=kinds, device=dev)
    abs_paths = [str(PHOEBI_ROOT / p) for p in paths]
    CACHE.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(PHOEBI_ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    stores = {}
    for k in kinds:
        p = store_path(k, tag)
        stores[k] = np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(len(paths), 16, ex.dim))
        p.with_suffix(".json").write_text(json.dumps({
            "kind": k, "backbone": backbone, "tile_size": 224, "grid": 4, "illumination": "divide/64",
            "l2_normalised": True, "n_images": len(paths), "paths_sha1": paths_digest(paths),
            "phoebi_commit": commit, "complete": False}, indent=2))
    done = 0
    for idxs, units in iter_image_units(ex, abs_paths, frame_batch_size, num_workers):
        ii = idxs.numpy()
        for k in kinds:
            stores[k][ii] = units[k].half().cpu().numpy()
        done += len(ii)
        if done % (frame_batch_size * 50) < frame_batch_size:
            print(f"  {done}/{len(paths)}", flush=True)
    for k in kinds:
        stores[k].flush()
        m = store_path(k, tag).with_suffix(".json")
        meta = json.loads(m.read_text()); meta["complete"] = True
        m.write_text(json.dumps(meta, indent=2))
