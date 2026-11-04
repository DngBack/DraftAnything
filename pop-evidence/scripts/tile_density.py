#!/usr/bin/env python3
"""Per-tile content proxy: std of the illumination-corrected grey level in each of the 16 grid tiles
(PHOEBI's correction and grid). Low std = mostly background. Unsupervised, so computed for every image.

    python scripts/tile_density.py --workers 32     # -> cache/tile_std.npy (N, 16) float32, Manifest order
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.features import CACHE  # noqa: E402
from popev.phoebi import PHOEBI_ROOT, Manifest  # noqa: E402


def tile_std(path: str) -> np.ndarray:
    sys.path.insert(0, str(PHOEBI_ROOT))
    from PIL import Image
    from src.common.illumination import normalize_illumination
    from src.common.tiling import _grid_offsets
    g = np.asarray(Image.open(PHOEBI_ROOT / path).convert("L"))
    g = normalize_illumination(g, sigma=64.0, method="divide").astype(np.float32)
    ys, xs = _grid_offsets(g.shape[0], 224, 4), _grid_offsets(g.shape[1], 224, 4)
    return np.array([g[y:y + 224, x:x + 224].std() for y in ys for x in xs], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    man = Manifest.default()
    with ProcessPoolExecutor(a.workers) as ex:
        out = np.stack(list(ex.map(tile_std, man.paths, chunksize=256)))
    np.save(CACHE / "tile_std.npy", out)
    print(out.shape, np.percentile(out, [5, 25, 50, 75, 95]).round(2))


if __name__ == "__main__":
    main()
