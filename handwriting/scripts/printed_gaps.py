"""Measure the printed prompt's word spacing on scanned IAM forms (exploratory confound check).

The prompt is typeset with full justification, so spaces vary by printed line and after
periods. Writers could copy that spacing. Output: one row per printed word boundary.
"""
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu

from hwline.segment import segment_words


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def process(png):
    form = png.stem
    root = ET.parse(f"data/iam/xml/{form}.xml").getroot()
    lines = [l.attrib["text"] for l in root.iter("machine-print-line")]
    g = np.asarray(Image.open(png).convert("L"))[:1400]
    ink = g < threshold_otsu(g)
    # A rule is one very long horizontal ink run; skew is absorbed by a 5-row union.
    long_run = np.zeros(ink.shape[0])
    spans = {}
    for y in range(ink.shape[0] - 5):
        r = runs(ink[y:y + 5].any(0))
        if r:
            best = max(r, key=lambda t: t[1] - t[0])
            long_run[y], spans[y] = best[1] - best[0], best
    hits = np.nonzero(long_run > 0.55 * ink.shape[1])[0]
    if len(hits) == 0:
        return [], "no_rules"
    groups = np.split(hits, np.nonzero(np.diff(hits) > 8)[0] + 1)
    rules = [int(h.mean()) for h in groups]
    xa, xb = spans[int(groups[0][0])]
    # Merge a rule detected twice, then require a plausible block height.
    rules = [r for i, r in enumerate(rules) if i == 0 or r - rules[i - 1] > 30]
    if len(rules) < 2 or rules[1] - rules[0] < 50:
        return [], "no_block"
    top, bot = rules[0] + 8, rules[1] - 8
    block = g[top:bot, max(xa - 5, 0):xb + 5]
    if block.size == 0 or block.min() == block.max():
        return [], "no_block"
    inkb = block < threshold_otsu(block)
    tl = [(a, b) for a, b in runs(inkb.mean(1) > 0.01) if b - a >= 12]
    if len(tl) != len(lines):
        return [], "line_count"
    rows = []
    for ln, (a, b), text in zip(range(len(lines)), tl, lines):
        toks = text.split()
        strip = block[max(a - 2, 0):b + 2]
        boxes, _, margin = segment_words(strip, len(toks))
        if boxes is None or len(toks) < 2:
            continue
        gaps = [boxes[i + 1][0] - boxes[i][2] for i in range(len(toks) - 1)]
        med = float(np.median(gaps))
        for i, gp in enumerate(gaps):
            rows.append({"form_id": form, "prompt": form.rstrip("abcdefghijklmnopqrstuvwxyz"), "pline": ln, "k": i,
                         "text_l": toks[i], "text_r": toks[i + 1], "print_gap_px": gp, "print_line_med": med,
                         "print_gap_rel": gp / max(med, 1), "seg_margin": margin})
    return rows, "ok"


def main():
    files = sorted(Path("data/iam/forms").glob("*.png"))
    with Pool(14) as p:
        res = p.map(process, files, chunksize=8)
    print(pd.Series([s for _, s in res]).value_counts().to_dict())
    df = pd.DataFrame([r for rs, _ in res for r in rs])
    # A prompt is printed identically on every form; average across scans.
    df["key"] = df.prompt + "|" + df.pline.astype(str) + "|" + df.k.astype(str)
    agg = df.groupby("key").agg(prompt=("prompt", "first"), pline=("pline", "first"), k=("k", "first"), text_l=("text_l", "first"), text_r=("text_r", "first"),
                                print_gap_px=("print_gap_px", "mean"), print_gap_rel=("print_gap_rel", "mean"),
                                scan_sd=("print_gap_px", "std"), n_scans=("print_gap_px", "size")).reset_index(drop=True)
    agg.to_csv("data/ling/printed_gaps.csv.gz", index=False)
    print(len(agg), agg.prompt.nunique())
    print(agg[["print_gap_px", "print_gap_rel", "scan_sd", "n_scans"]].describe().round(2).T.to_string())


if __name__ == "__main__":
    main()
