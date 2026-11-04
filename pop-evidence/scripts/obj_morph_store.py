#!/usr/bin/env python3
"""Object embeddings + explicit morphology: append standardised log-area, log-contrast and elongation of
each detected object to its DINOv2 embedding (crops are resized to a fixed input, so absolute size is
otherwise only implicit). Standardisation statistics come from the LCO training images only.

    python scripts/obj_morph_store.py      # -> cache/features_cls_objmorph.npy (N, 48, 384 + 3)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.features import CACHE, open_store, paths_digest, store_path  # noqa: E402
from popev.phoebi import Manifest  # noqa: E402

man = Manifest.default()
X = open_store(man.paths, "cls", "_obj")
meta = np.load(CACHE / "objects_meta_obj.npz")["meta"]                  # (N, 48, 5): area, contrast, elong, cy, cx
m = np.stack([np.log1p(meta[..., 0]), np.log1p(meta[..., 1]), np.log(np.maximum(meta[..., 2], 1.0))], -1)
tr = man.parts["train"].rows
mu, sd = m[tr].reshape(-1, 3).mean(0), m[tr].reshape(-1, 3).std(0) + 1e-6
m = ((m - mu) / sd) * 0.1      # scaled to the per-dimension magnitude of L2-normalised 384-d embeddings
out = np.lib.format.open_memmap(store_path("cls", "_objmorph"), mode="w+", dtype=np.float16, shape=(X.shape[0], X.shape[1], X.shape[2] + 3))
for s in range(0, X.shape[0], 5000):
    out[s:s + 5000] = np.concatenate([np.asarray(X[s:s + 5000], dtype=np.float32), m[s:s + 5000]], -1).astype(np.float16)
out.flush()
store_path("cls", "_objmorph").with_suffix(".json").write_text(json.dumps({
    "kind": "cls", "units": "objects+morphology", "paths_sha1": paths_digest(man.paths), "complete": True,
    "morph_mu": mu.tolist(), "morph_sd": sd.tolist(), "n_images": X.shape[0]}))
print("wrote", out.shape)
