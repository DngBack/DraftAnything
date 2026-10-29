#!/usr/bin/env python3
"""Object-centric features for every PHOEBI image: detect objects (popev.objects), crop 64 px around each,
embed with frozen DINOv2-S/14 at 112 px. Output in Manifest order, same layout as the tile store:

    cache/features_cls_obj.npy  (N, 48, 384) fp16   object embeddings (L2-normalised); images with fewer
                                                     than 48 objects repeat theirs cyclically
    cache/objects_meta.npz      n_valid (N,), area/contrast/elong/cy/cx (N, 48)

    python scripts/extract_objects.py --workers 32 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.features import CACHE, paths_digest, store_path  # noqa: E402
from popev.phoebi import PHOEBI_ROOT, Manifest  # noqa: E402

M, CROP, INP = 48, 64, 112


def work(path: str):
    sys.path.insert(0, str(PHOEBI_ROOT))
    from PIL import Image
    from src.common.illumination import normalize_illumination
    from popev.objects import crops, detect
    g = normalize_illumination(np.asarray(Image.open(PHOEBI_ROOT / path).convert("L")), sigma=64.0, method="divide")
    objs = detect(g, M)
    n = len(objs)
    if n == 0:                                   # empty field: one centre crop stands in, flagged n_valid = 0
        objs = [{"cy": g.shape[0] / 2, "cx": g.shape[1] / 2, "area": 0, "contrast": 0, "elong": 1}]
    sel = [objs[i % len(objs)] for i in range(M)]
    cr = crops(np.repeat(g[..., None], 3, 2), sel, CROP)
    meta = np.array([[o["area"], o["contrast"], o["elong"], o["cy"], o["cx"]] for o in sel], dtype=np.float32)
    return cr, meta, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import timm
    import torch
    import torch.nn.functional as F
    man = Manifest.default(); paths = man.paths[: a.limit] if a.limit else man.paths
    dev = torch.device(a.device)
    model = timm.create_model(a.backbone, pretrained=True, img_size=INP, num_classes=0).to(dev).eval()
    mean = torch.tensor(model.default_cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(model.default_cfg["std"], device=dev).view(1, 3, 1, 1)
    tag = "_obj" if not a.limit else "_objsmoke"
    Fm = np.lib.format.open_memmap(store_path("cls", tag), mode="w+", dtype=np.float16, shape=(len(paths), M, model.num_features))
    meta = np.zeros((len(paths), M, 5), np.float32); nval = np.zeros(len(paths), np.int32)
    t0 = time.time(); buf, idx = [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).to(dev).float().view(-1, CROP, CROP, 3).permute(0, 3, 1, 2) / 255.0
        x = F.interpolate(x, size=INP, mode="bilinear", align_corners=False)
        with torch.no_grad(), torch.autocast("cuda" if dev.type == "cuda" else "cpu"):
            z = model((x - mean) / std)
        Fm[idx] = F.normalize(z.float(), dim=-1).view(len(idx), M, -1).cpu().numpy()
        buf.clear(); idx.clear()

    with Pool(a.workers) as pool:
        for i, (cr, mt, n) in enumerate(pool.imap(work, paths, chunksize=16)):
            buf.append(cr); idx.append(i); meta[i] = mt; nval[i] = n
            if len(buf) == 64:
                flush()
            if i % 5000 == 0:
                print(f"  {i}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
        if buf:
            flush()
    Fm.flush()
    np.savez_compressed(CACHE / f"objects_meta{tag}.npz", meta=meta, n_valid=nval)
    store_path("cls", tag).with_suffix(".json").write_text(json.dumps({
        "kind": "cls", "units": "objects", "max_objects": M, "crop": CROP, "input": INP, "backbone": a.backbone,
        "paths_sha1": paths_digest(paths), "complete": True, "n_images": len(paths)}))
    print(f"done {len(paths)} images in {time.time() - t0:.0f}s; median objects/image {np.median(nval):.0f}, "
          f"empty {int((nval == 0).sum())}")


if __name__ == "__main__":
    main()
