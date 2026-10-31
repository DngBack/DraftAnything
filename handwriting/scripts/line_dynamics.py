"""Measure and profile within-line dynamics for any source of line images.

  measure:  --list CSV(image_path, transcript, writer_id, line_id) --source NAME
            [--height 32]  -> words table (segmentation by column gaps, same for all sources)
  profile:  --words W1 W2 ... -> per-source, per-feature dynamics profile
  segcheck: compare gap segmentation with IAM XML word boxes on real lines
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from hwline.profile import dynamics_profile, measure_line
from hwline.segment import segment_words, written_tokens


def load_gray(path, height):
    im = Image.open(path).convert("L")
    if height:
        w = max(int(round(im.width * height / im.height)), 1)
        im = im.resize((w, height), Image.BILINEAR)
    return np.asarray(im)


def _measure(args):
    rec, height, source = args
    gray = load_gray(rec["image_path"], height)
    meta = {"source": source, "line_id": rec["line_id"], "writer_id": str(rec["writer_id"])}
    rows, ok, margin = measure_line(gray, rec["transcript"], meta)
    return rows, ok, margin


def cmd_measure(a):
    lst = pd.read_csv(a.list, dtype={"writer_id": str})
    with Pool(a.workers) as p:
        res = p.map(_measure, [(r, a.height, a.source) for r in lst.to_dict("records")], chunksize=8)
    rows = [r for rs, _, _ in res for r in rs]
    ok = np.array([o for _, o, _ in res])
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(json.dumps({"source": a.source, "lines": len(lst), "segmented": int(ok.sum()),
                      "coverage": round(float(ok.mean()), 4), "words": len(rows)}))


def cmd_profile(a):
    allrows = []
    for w in a.words:
        df = pd.read_csv(w, dtype={"writer_id": str})
        for r in dynamics_profile(df, a.n_perm, a.n_boot, a.seed):
            allrows.append({"source": df.source.iloc[0], **r})
            print(df.source.iloc[0], r["feature"], {k: round(v, 3) for k, v in r.items()
                                                     if isinstance(v, float)}, flush=True)
    pd.DataFrame(allrows).to_csv(a.out, index=False)


def _segcheck(args):
    rec, height = args
    gray = load_gray(rec["image_path"], height)
    scale = gray.shape[0] / rec["orig_height"]
    texts, boxes = json.loads(rec["word_texts"]), json.loads(rec["word_boxes"])
    # Merge punctuation tokens' XML boxes into the previous word, as written_tokens does.
    merged_t, merged_b = [], []
    for t, b in zip(texts, boxes):
        if merged_t and not any(c.isalnum() for c in t):
            merged_t[-1] += t
            pb = merged_b[-1]
            merged_b[-1] = [min(pb[0], b[0]), min(pb[1], b[1]), max(pb[2], b[2]), max(pb[3], b[3])]
        else:
            merged_t.append(t)
            merged_b.append(list(b))
    toks = written_tokens(rec["transcript"])
    seg, _, margin = segment_words(gray, len(toks))
    if seg is None or len(merged_b) != len(seg):
        return {"ok": False, "margin": margin, "all_match": False, "n": len(toks), "ious": []}
    ious = []
    for (s0, _, s1, _), b in zip(seg, merged_b):
        b0, b1 = b[0] * scale, b[2] * scale
        inter = max(0, min(s1, b1) - max(s0, b0))
        ious.append(inter / max(max(s1, b1) - min(s0, b0), 1e-9))
    return {"ok": True, "margin": margin, "all_match": bool(min(ious) >= 0.7), "n": len(toks), "ious": ious}


def cmd_segcheck(a):
    man = pd.read_csv("data/manifest.csv", dtype={"writer_id": str})
    sub = man[(man.split == a.split) & (man.quality_flag == "ok")]
    with Pool(a.workers) as p:
        res = p.map(_segcheck, [(r, a.height) for r in sub.to_dict("records")], chunksize=8)
    ok = np.array([r["ok"] for r in res]); am = np.array([r["all_match"] for r in res])
    ious = np.concatenate([r["ious"] for r in res if r["ious"]])
    out = {"split": a.split, "height": a.height or "orig", "lines": len(res),
           "segmented": round(float(ok.mean()), 4), "all_words_iou>=0.7": round(float(am.mean()), 4),
           "word_iou_median": round(float(np.median(ious)), 3),
           "word_iou>=0.7": round(float((ious >= 0.7).mean()), 4)}
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("measure")
    m.add_argument("--list", required=True); m.add_argument("--source", required=True)
    m.add_argument("--height", type=int, default=0); m.add_argument("--out", required=True)
    m.add_argument("--workers", type=int, default=32)
    p = sp.add_parser("profile")
    p.add_argument("--words", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n-perm", type=int, default=200); p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("segcheck")
    s.add_argument("--split", default="val1"); s.add_argument("--height", type=int, default=0)
    s.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    {"measure": cmd_measure, "profile": cmd_profile, "segcheck": cmd_segcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
