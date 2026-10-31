"""Per-word descriptors on original-resolution line images (plan §4.1).

Only train and val1 are processed: val2/test are not used for direction decisions.
Writes data/words_{split}.csv.gz with one row per IAM word token.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from hwline.descriptors import content_features, word_descriptors


def process(rec: dict) -> list[dict]:
    gray = np.asarray(Image.open(rec["image_path"]).convert("L"))
    texts, boxes = json.loads(rec["word_texts"]), json.loads(rec["word_boxes"])
    out = []
    for i, (t, b) in enumerate(zip(texts, boxes)):
        row = {"line_id": rec["line_id"], "form_id": rec["form_id"], "writer_id": rec["writer_id"],
               "split": rec["split"], "pos": i, "n_tokens": len(texts), "text": t,
               "line_h": gray.shape[0], "line_w": gray.shape[1]}
        if b is None or b[2] - b[0] < 2 or b[3] - b[1] < 2:
            row["n_ink"] = 0
        else:
            row["box_x0"], row["box_y0"], row["box_x1"], row["box_y1"] = b
            row.update(word_descriptors(gray, tuple(b), rec["threshold"], t))
        row.update(content_features(t))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--splits", nargs="+", default=["train", "val1"])
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    man = pd.read_csv(a.manifest, dtype={"writer_id": str})
    for s in a.splits:
        sub = man[(man.split == s) & (man.quality_flag == "ok")]
        with Pool(a.workers) as p:
            rows = [r for rs in p.imap(process, sub.to_dict("records"), chunksize=16) for r in rs]
        df = pd.DataFrame(rows)
        df.to_csv(f"data/words_{s}.csv.gz", index=False)
        v = df[[c for c in df.columns if c.startswith("v_")]].mean()
        print(s, "lines", sub.shape[0], "excluded(flagged)", int(((man.split == s) & (man.quality_flag != "ok")).sum()),
              "words", len(df), "valid rates", v.round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
