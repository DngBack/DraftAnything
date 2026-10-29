#!/usr/bin/env python3
"""Exploratory (post-hoc on phoebi4, which has been read): per-batch feature standardisation.
phoebi6 rows are standardised (per dimension, over all tiles) with statistics of the phoebi6 training images;
phoebi4 rows with statistics of all phoebi4 images (unlabelled, transductive at the batch level). Statistics are
pooled over every video of a batch, so species signal is not removed the way per-video centring would.
    python scripts/batch_norm_store.py   # -> cache/features_cls_p4bn.npy
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from popev.features import open_store, paths_digest, store_path
from popev.phoebi import Manifest
paths = json.loads((ROOT / "data/phoebi4/store_paths.json").read_text())
man = Manifest(ROOT / "data/phoebi4/splits_lco.json", store_paths=paths)
F = open_store(man.paths, "cls", "_p4")
tr = np.sort(man.parts["train"].rows); te = np.sort(man.parts["test"].rows)
six = np.arange(len(paths)) < te.min()
def stats(rows):
    x = np.asarray(F[rows[:: max(1, len(rows) // 20000)]], dtype=np.float32).reshape(-1, F.shape[-1])
    return x.mean(0), x.std(0) + 1e-4
m6, s6 = stats(tr); m4, s4 = stats(te)
out = store_path("cls", "_p4bn")
S = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=F.shape)
for i in range(0, len(paths), 4096):
    blk = np.asarray(F[i:i + 4096], dtype=np.float32); is6 = six[i:i + 4096]
    blk[is6] = (blk[is6] - m6) / s6; blk[~is6] = (blk[~is6] - m4) / s4
    blk /= np.linalg.norm(blk, axis=-1, keepdims=True)              # keep unit norm like the original store
    S[i:i + 4096] = blk.astype(np.float16)
S.flush()
out.with_suffix(".json").write_text(json.dumps({"kind": "cls", "paths_sha1": paths_digest(paths), "complete": True,
                                                "note": "per-batch standardised (exploratory)"}))
print("shift of batch means (cosine of mean vectors):", float(m6 @ m4 / np.linalg.norm(m6) / np.linalg.norm(m4)))
