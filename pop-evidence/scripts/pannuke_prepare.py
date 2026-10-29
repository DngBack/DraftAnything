#!/usr/bin/env python3
"""PanNuke (H&E patches, CC BY-NC-SA 4.0) in the PopEvidence format, with instance ground truth.

Classes (mask channels): neoplastic, inflammatory, connective, dead, epithelial; background dropped.
A class is present in a 256 px patch if it has at least one instance. Tiles: 4x4 grid of 64 px,
resized to 224 for DINOv2-S/14. Tile ground truth: an instance centroid of class k lies in the tile.

Unit = image (each patch is its own "combo"). Official split: train on fold 1, calibrate on fold 2,
evaluate on fold 3 (one "fold" entry); fold 3 is also stored as `test` for a later locked run, and the
other two rotations of the official protocol are written as folds 1 and 2.

    python scripts/pannuke_prepare.py --device cuda
Outputs: data/pannuke/{splits_lco.json, lco_dev_folds.json}, cache/features_cls_pannuke.npy (N, 16, 384),
cache/tile_labels_pannuke.npy (N, 16, 5), cache/instance_counts_pannuke.npy (N, 5), data/pannuke/tissue.npy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.features import CACHE, paths_digest, store_path  # noqa: E402

SRC = Path("/home/jovyan/bachdx2/datasets/pannuke/raw")
K = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]
G, TS = 4, 64


def find(fold: int, name: str) -> Path:
    hits = sorted(SRC.rglob(f"*{fold}*/**/{name}")) or sorted(p for p in SRC.rglob(name) if f"{fold}" in str(p))
    if not hits:
        raise SystemExit(f"{name} for fold {fold} not found under {SRC}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    a = ap.parse_args()
    imgs, labs, TL, IC, tissue, ids = [], [], [], [], [], []
    for fold in (1, 2, 3):
        X = np.load(find(fold, "images.npy"), mmap_mode="r")
        M = np.load(find(fold, "masks.npy"), mmap_mode="r")
        T = np.load(find(fold, "types.npy"))
        print(f"fold {fold}: images {X.shape}, masks {M.shape}", flush=True)
        for i in range(len(X)):
            m = np.asarray(M[i])
            tl = np.zeros((G * G, len(K)), np.uint8); ic = np.zeros(len(K), np.int32)
            for k in range(len(K)):
                inst = m[..., k].astype(np.int64)
                for v in np.unique(inst):
                    if v == 0:
                        continue
                    ys, xs = np.nonzero(inst == v)
                    cy, cx = ys.mean(), xs.mean(); ic[k] += 1
                    tl[min(int(cy // TS), G - 1) * G + min(int(cx // TS), G - 1), k] = 1
            imgs.append((fold, i)); TL.append(tl); IC.append(ic); labs.append((ic > 0).astype(int).tolist())
            tissue.append(str(T[i])); ids.append(f"f{fold}_{i:05d}")
    TL, IC = np.stack(TL), np.stack(IC)
    folds_of = np.array([f for f, _ in imgs])
    out = ROOT / "data/pannuke"; out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    order = []
    for j, (f, i) in enumerate(imgs):
        part = "test" if f == 3 else "train"
        splits[part].append({"path": ids[j], "label": labs[j], "combo": ids[j],
                             "classes": "_".join(k for k, v in zip(K, labs[j]) if v) or "none"})
    order = [e["path"] for e in splits["train"]] + [e["path"] for e in splits["test"]]
    pos = {p: j for j, p in enumerate(ids)}; perm = np.array([pos[p] for p in order])
    (out / "splits_lco.json").write_text(json.dumps({
        "class_names": K, "heldout_combos": [e["combo"] for e in splits["test"]],
        "trained_combos": [e["combo"] for e in splits["train"]], "splits": splits,
        "note": "unit = image; official PanNuke folds; see scripts/pannuke_prepare.py"}))
    f1 = [p for p in ids if p.startswith("f1_")]; f2 = [p for p in ids if p.startswith("f2_")]
    # development uses folds 1 and 2 only: train on one, calibrate on half of the other, evaluate on the rest
    h1, h2 = f1[: len(f1) // 2], f2[: len(f2) // 2]
    folds = [{"fold": 0, "train_combos": f1, "calib_combos": h2, "dev_combos": f2[len(f2) // 2:]},
             {"fold": 1, "train_combos": f2, "calib_combos": h1, "dev_combos": f1[len(f1) // 2:]}]
    (out / "lco_dev_folds.json").write_text(json.dumps({"folds": folds}))
    np.save(out / "tissue.npy", np.array(tissue)[perm])
    np.save(CACHE / "tile_labels_pannuke.npy", TL[perm]); np.save(CACHE / "instance_counts_pannuke.npy", IC[perm])

    import timm
    import torch
    import torch.nn.functional as F
    dev = torch.device(a.device)
    model = timm.create_model(a.backbone, pretrained=True, img_size=224, num_classes=0).to(dev).eval()
    mean = torch.tensor(model.default_cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(model.default_cfg["std"], device=dev).view(1, 3, 1, 1)
    Fm = np.lib.format.open_memmap(store_path("cls", "_pannuke"), mode="w+", dtype=np.float16,
                                   shape=(len(order), G * G, model.num_features))
    cache = {f: np.load(find(f, "images.npy"), mmap_mode="r") for f in (1, 2, 3)}
    B = 64
    with torch.no_grad():
        for s in range(0, len(order), B):
            idx = perm[s:s + B]
            batch = np.stack([np.asarray(cache[imgs[j][0]][imgs[j][1]]) for j in idx]).astype(np.float32)
            x = torch.from_numpy(batch).to(dev).permute(0, 3, 1, 2) / 255.0
            if x.max() > 1.5:          # some releases store 0-255 floats, guard against double scaling
                x = x / 255.0
            t = x.unfold(2, TS, TS).unfold(3, TS, TS)                     # (b, 3, G, G, TS, TS)
            t = t.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, TS, TS)
            t = F.interpolate(t, size=224, mode="bilinear", align_corners=False)
            with torch.autocast("cuda" if dev.type == "cuda" else "cpu"):
                z = model((t - mean) / std)
            Fm[s:s + len(idx)] = F.normalize(z.float(), dim=-1).view(len(idx), G * G, -1).cpu().numpy()
            if s % (B * 20) == 0:
                print(f"  {s}/{len(order)}", flush=True)
    Fm.flush()
    store_path("cls", "_pannuke").with_suffix(".json").write_text(json.dumps({
        "kind": "cls", "dataset": "PanNuke", "backbone": a.backbone, "grid": G, "tile_px": TS,
        "paths_sha1": paths_digest(order), "complete": True, "n_images": len(order)}))
    Y = np.array([e["label"] for e in splits["train"]])
    print(f"{len(order)} images; train-pool prevalence {dict(zip(K, Y.mean(0).round(3)))}; "
          f"distinct label sets {len({tuple(r) for r in Y})}")


if __name__ == "__main__":
    main()
