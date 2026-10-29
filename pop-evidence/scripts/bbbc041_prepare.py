#!/usr/bin/env python3
"""BBBC041 (P. vivax blood smears, CC BY-NC-SA 3.0) in the PopEvidence format.

Classes: gametocyte, leukocyte, ring, schizont, trophozoite (red blood cells are in every image and
are dropped as a class). Images with any "difficult" object are dropped (ambiguous labels).

The grouping unit is the image: each image is its own "combo", so bags are single images, the
bootstrap resamples images, and the own/context regression has one point per image. Folds: 5-fold
CV over the official training images (dev = 1/5; calibration = 1/5 of the rest; train = remainder).
The official test images (another acquisition, 1944x1383) are the `test` part, used once.

Also writes tile-level ground truth: a tile contains class k if the centre of a k box lies in it.

    python scripts/bbbc041_prepare.py --device cuda
Outputs: data/bbbc041/{splits_lco.json, lco_dev_folds.json}, cache/features_cls_bbbc041.npy (N, 35, 384),
cache/tile_labels_bbbc041.npy (N, 35, 5), cache/instance_counts_bbbc041.npy (N, 5).
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

SRC = Path("/home/jovyan/bachdx2/datasets/bbbc041/raw/malaria")
K = ["gametocyte", "leukocyte", "ring", "schizont", "trophozoite"]
TILE, ROWS, COLS = 224, 5, 7


def offsets(dim: int, n: int):
    span = dim - TILE
    return [round(i * span / (n - 1)) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    entries = {"train": [], "test": []}
    for part, f in (("train", "training.json"), ("test", "test.json")):
        for e in json.loads((SRC / f).read_text()):
            cats = {o["category"] for o in e["objects"]}
            if "difficult" in cats:
                continue
            entries[part].append(e)
    rng = np.random.default_rng(a.seed)

    # manifests: every image is its own unit
    out_dir = ROOT / "data/bbbc041"; out_dir.mkdir(parents=True, exist_ok=True)
    rows, paths = [], []
    splits = {"train": [], "val": [], "test": []}
    for part in ("train", "test"):
        for e in entries[part]:
            p = str(SRC / e["image"]["pathname"].lstrip("/"))
            uid = Path(p).stem
            lab = [int(any(o["category"] == k for o in e["objects"])) for k in K]
            splits[part].append({"path": p, "label": lab, "combo": uid,
                                 "classes": "_".join(k for k, v in zip(K, lab) if v) or "none"})
            rows.append(e); paths.append(p)
    paths = [x["path"] for x in splits["train"]] + [x["path"] for x in splits["test"]]   # Manifest order
    d = {"class_names": K, "heldout_combos": [x["combo"] for x in splits["test"]],
         "trained_combos": [x["combo"] for x in splits["train"]], "splits": splits,
         "note": "unit = image; see scripts/bbbc041_prepare.py"}
    (out_dir / "splits_lco.json").write_text(json.dumps(d))
    ids = np.array(d["trained_combos"]); perm = ids[rng.permutation(len(ids))]
    chunks = np.array_split(perm, 5); folds = []
    for f in range(5):
        dev = chunks[f]; rest = np.concatenate([chunks[g] for g in range(5) if g != f])
        cal, tr = rest[: len(rest) // 5], rest[len(rest) // 5:]
        folds.append({"fold": f, "dev_combos": sorted(dev.tolist()), "calib_combos": sorted(cal.tolist()),
                      "train_combos": sorted(tr.tolist())})
    (out_dir / "lco_dev_folds.json").write_text(json.dumps({"folds": folds}))

    # tile ground truth and instance counts
    by_path = {str(SRC / e["image"]["pathname"].lstrip("/")): e for e in rows}
    TL = np.zeros((len(paths), ROWS * COLS, len(K)), np.uint8); IC = np.zeros((len(paths), len(K)), np.int32)
    for i, p in enumerate(paths):
        e = by_path[p]; H, W = e["image"]["shape"]["r"], e["image"]["shape"]["c"]
        ys, xs = offsets(H, ROWS), offsets(W, COLS)
        for o in e["objects"]:
            if o["category"] not in K:
                continue
            k = K.index(o["category"]); IC[i, k] += 1
            bb = o["bounding_box"]; cy = (bb["minimum"]["r"] + bb["maximum"]["r"]) / 2; cx = (bb["minimum"]["c"] + bb["maximum"]["c"]) / 2
            for t, (y, x) in enumerate((y, x) for y in ys for x in xs):
                if y <= cy < y + TILE and x <= cx < x + TILE:
                    TL[i, t, k] = 1
    np.save(CACHE / "tile_labels_bbbc041.npy", TL); np.save(CACHE / "instance_counts_bbbc041.npy", IC)

    # features: plain ImageNet normalisation (brightfield RGB, no illumination correction)
    import timm
    import torch
    from PIL import Image
    dev = torch.device(a.device)
    model = timm.create_model(a.backbone, pretrained=True, img_size=TILE, num_classes=0).to(dev).eval()
    mean = torch.tensor(model.default_cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(model.default_cfg["std"], device=dev).view(1, 3, 1, 1)
    F = np.lib.format.open_memmap(store_path("cls", "_bbbc041"), mode="w+", dtype=np.float16,
                                  shape=(len(paths), ROWS * COLS, model.num_features))
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).to(dev).permute(2, 0, 1)[None].float() / 255
            H, W = img.shape[-2:]
            tiles = torch.cat([img[..., y:y + TILE, x:x + TILE] for y in offsets(H, ROWS) for x in offsets(W, COLS)])
            with torch.autocast("cuda" if dev.type == "cuda" else "cpu"):
                z = model((tiles - mean) / std)
            F[i] = torch.nn.functional.normalize(z.float(), dim=-1).cpu().numpy()
            if i % 200 == 0:
                print(f"  {i}/{len(paths)}", flush=True)
    F.flush()
    store_path("cls", "_bbbc041").with_suffix(".json").write_text(json.dumps({
        "kind": "cls", "dataset": "BBBC041", "backbone": a.backbone, "grid": [ROWS, COLS], "tile": TILE,
        "paths_sha1": paths_digest(paths), "complete": True, "n_images": len(paths)}))
    tl = TL[: len(splits["train"])]
    print(f"{len(paths)} images ({len(splits['train'])} train, {len(splits['test'])} test); "
          f"tiles containing each class (train): {dict(zip(K, tl.reshape(-1, len(K)).mean(0).round(4)))}")


if __name__ == "__main__":
    main()
