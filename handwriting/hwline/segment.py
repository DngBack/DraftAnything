"""Source-agnostic word segmentation for line images (real or generated).

Generated lines have no word boxes, so every source (including real lines, for a fair
comparison) is segmented the same way: binarise (Otsu), find runs of ink-free columns and
cut at the n_words-1 widest runs, where n_words comes from the transcript. A margin
statistic says how clearly the chosen gaps beat the best rejected one.
"""
from __future__ import annotations

import re

import numpy as np
from skimage.filters import threshold_otsu

PUNCT_TOKEN = re.compile(r"^[^\w]+$")


def written_tokens(transcript: str) -> list[str]:
    """IAM transcripts separate punctuation with spaces ('Exchange .'); on paper it is attached
    to the previous word, so merge punctuation-only tokens into the preceding token."""
    out: list[str] = []
    for t in transcript.split():
        if out and PUNCT_TOKEN.match(t):
            out[-1] += t
        else:
            out.append(t)
    return out


def otsu_ink(gray: np.ndarray) -> tuple[np.ndarray, int]:
    if gray.max() == gray.min():
        return np.zeros_like(gray, bool), int(gray.max())
    t = threshold_otsu(gray)
    return gray < t, int(t)


def gap_runs(ink: np.ndarray, min_col_ink: int = 0) -> list[tuple[int, int]]:
    """Runs [a, b) of columns with <= min_col_ink ink pixels, strictly inside the ink extent."""
    col = ink.sum(0)
    on = np.nonzero(col > min_col_ink)[0]
    if on.size == 0:
        return []
    lo, hi = on[0], on[-1] + 1
    empty = col[lo:hi] <= min_col_ink
    runs, start = [], None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            runs.append((lo + start, lo + i))
            start = None
    return runs


def segment_words(gray: np.ndarray, n_words: int, min_ink_frac: float = 0.0):
    """Returns (boxes [(x0,y0,x1,y1)], threshold, margin) or (None, threshold, 0) on failure.
    margin = (narrowest chosen gap - widest rejected gap) / median ink-height of the line."""
    ink, thr = otsu_ink(gray)
    # Ignore isolated specks: a column counts as ink only if it has more than a few pixels.
    min_col = max(int(min_ink_frac * gray.shape[0]), 0)
    runs = gap_runs(ink, min_col)
    if n_words < 1 or len(runs) < n_words - 1:
        return None, thr, 0.0
    widths = np.array([b - a for a, b in runs])
    order = np.argsort(-widths, kind="stable")
    chosen = sorted(order[: n_words - 1])
    rows = np.nonzero(ink.any(1))[0]
    scale = max(rows[-1] - rows[0] + 1, 1) if rows.size else 1
    rejected = widths[order[n_words - 1]] if len(runs) > n_words - 1 else 0
    margin = ((widths[order[n_words - 2]] if n_words > 1 else scale) - rejected) / scale
    col = ink.sum(0)
    on = np.nonzero(col > min_col)[0]
    cuts = [on[0]] + [x for i in chosen for x in runs[i]] + [on[-1] + 1]
    boxes = []
    for k in range(n_words):
        x0, x1 = cuts[2 * k], cuts[2 * k + 1]
        r = np.nonzero(ink[:, x0:x1].any(1))[0]
        if r.size == 0:
            return None, thr, 0.0
        boxes.append((int(x0), int(r[0]), int(x1), int(r[-1] + 1)))
    return boxes, thr, float(margin)
