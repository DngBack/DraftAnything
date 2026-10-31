"""Word HDF5 in SpiS-GAN's format, restricted to our Large Writer Independent split.

The public SpiS-GAN checkpoint was trained on a 339-writer split that contains 36 of our
test writers (reports/audit/spis_writer_map_train.csv), so it cannot be used for init or
evaluation. This builds clean word data: crops from the IAM line images with XML word
boxes, background above the line's IAM threshold set to white, height 32.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/home/jovyan/bachdx2/repos/spis-gan")
from lib.alphabet import Alphabets  # noqa: E402


def crops(man_split, alpha, max_len):
    for r in man_split.itertuples():
        g = np.asarray(Image.open(r.image_path).convert("L"))
        for t, b in zip(json.loads(r.word_texts), json.loads(r.word_boxes)):
            if not b or not t or len(t) > max_len or any(c not in alpha for c in t):
                continue
            x0, y0, x1, y1 = b
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            c = g[y0:y1, x0:x1].copy()
            c[c >= r.threshold] = 255
            w = max(int(round(c.shape[1] * 32 / c.shape[0])), 1)
            yield np.asarray(Image.fromarray(c).resize((w, 32), Image.BILINEAR)), t, r.writer_id


def write(path, items, wid_map):
    imgs, lbs, iseek, ilen, lseek, llen, wids = [], [], [], [], [], [], []
    ip = lp = 0
    for im, t, w in items:
        imgs.append(im); iseek.append(ip); ilen.append(im.shape[1]); ip += im.shape[1]
        codes = [ord(c) for c in t]
        lbs.extend(codes); lseek.append(lp); llen.append(len(codes)); lp += len(codes)
        wids.append(wid_map[w])
    with h5py.File(path, "w") as f:
        f["imgs"] = np.concatenate(imgs, 1).astype(np.uint8)
        f["lbs"] = np.array(lbs, np.int32)
        f["img_seek_idxs"] = np.array(iseek, np.int64); f["img_lens"] = np.array(ilen, np.int64)
        f["lb_seek_idxs"] = np.array(lseek, np.int64); f["lb_lens"] = np.array(llen, np.int64)
        f["wids"] = np.array(wids, np.int64)
    return len(ilen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/home/jovyan/bachdx2/datasets/spis_clean"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv("data/manifest.csv", dtype={"writer_id": str})
    man = man[man.quality_flag == "ok"]
    alpha = set(Alphabets["all"]) - {"`"}
    tr = man[man.split == "train"]
    wid_map = {w: i for i, w in enumerate(sorted(tr.writer_id.unique()))}
    n_tr = write(a.out / "train_32.hdf5", crops(tr, alpha, 20), wid_map)
    va = man[man.split == "val1"]
    vmap = {w: i for i, w in enumerate(sorted(va.writer_id.unique()))}
    n_va = write(a.out / "test_32.hdf5", crops(va, alpha, 20), vmap)
    json.dump({"train_writers": wid_map, "val1_writers": vmap}, open(a.out / "writer_maps.json", "w"))
    print({"train_words": n_tr, "train_writers": len(wid_map), "val1_words": n_va, "val1_writers": len(vmap)})


if __name__ == "__main__":
    main()
