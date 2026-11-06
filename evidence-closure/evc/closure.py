"""Heuristic relational evidence closure on OCR layout + token-set selectors.

Nothing here reads the question or the gold answer: closure is a function of the
seed words and the page layout only.
"""
from __future__ import annotations

import re

import numpy as np

from .data import is_num

YEAR_RE = re.compile(r"^\(?(19|20)\d\d\)?[,:]?$")
SCALE_RE = re.compile(r"(thousand|million|billion|percent\b|\$'?000|\$m\b|\$bn\b|'000)", re.I)
ALPHA_RE = re.compile(r"[A-Za-z]")
_LAYOUT = {}


def _vov(a, b):
    """vertical overlap ratio of two boxes w.r.t. the smaller height"""
    o = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[3] - a[1], b[3] - b[1])
    return o / h if h > 0 else 0.0


def _is_header(t: str) -> bool:
    return bool(ALPHA_RE.search(t)) or bool(YEAR_RE.match(t))


def layout(words, page_w):
    """Split OCR blocks into visual lines; flag sentence-like lines (paragraph text)."""
    key = id(words)
    if key in _LAYOUT and _LAYOUT[key][0] is words:
        return _LAYOUT[key][1]
    by_block = {}
    for i, w in enumerate(words):
        by_block.setdefault(w["block"], []).append(i)
    lines = []
    for ids in by_block.values():
        cur = []
        for i in sorted(ids, key=lambda i: (words[i]["bbox"][1] + words[i]["bbox"][3]) / 2):
            for ln in cur:
                if _vov(words[ln[0]]["bbox"], words[i]["bbox"]) >= 0.5:
                    ln.append(i)
                    break
            else:
                cur.append([i])
        lines += cur
    line_of, para = {}, set()
    for li, ln in enumerate(lines):
        ln.sort(key=lambda i: words[i]["bbox"][0])
        for i in ln:
            line_of[i] = li
        x0 = min(words[i]["bbox"][0] for i in ln)
        x1 = max(words[i]["bbox"][2] for i in ln)
        n_alpha = sum(bool(ALPHA_RE.search(words[i]["text"])) for i in ln)
        gaps = [words[b]["bbox"][0] - words[a]["bbox"][2] for a, b in zip(ln, ln[1:])]
        if n_alpha >= 6 and max(gaps, default=0) < 40:
            para.add(li)
    out = (lines, line_of, para)
    _LAYOUT[key] = (words, out)
    return out


def _phrase(words, line, i):
    """Extend a header hit to adjacent header-like words on its line (gap <= 14 px)."""
    k = line.index(i)
    out = [i]
    for step in (-1, 1):
        j = k
        while 0 <= j + step < len(line):
            a, b = words[line[j]]["bbox"], words[line[j + step]]["bbox"]
            gap = b[0] - a[2] if step == 1 else a[0] - b[2]
            if gap > 14 or not _is_header(words[line[j + step]]["text"]):
                break
            j += step
            out.append(line[j])
    return out


def relations(words, wid, page_w):
    """Return dict(row=[ids], col=[ids], unit=[ids]) for numeric seed word `wid`."""
    lines, line_of, para = layout(words, page_w)
    w = words[wid]["bbox"]
    row, col, unit = [], [], []
    same_line = [i for i, v in enumerate(words) if i != wid and _vov(v["bbox"], w) >= 0.5]
    for i in same_line:
        v = words[i]
        if v["bbox"][2] <= w[0] + 2 and ALPHA_RE.search(v["text"]):
            row.append(i)
        t = v["text"].strip()
        # unit symbol glued to the value, e.g. "43 %" or "$ 3,123"
        if 0 <= v["bbox"][0] - w[2] <= 25 and "%" in t and not ALPHA_RE.search(t):
            unit.append(i)
        if 0 <= w[0] - v["bbox"][2] <= 25 and t in {"$", "€", "£", "RMB", "US$"}:
            unit.append(i)
    # column headers: lines above, x-overlapping, header-like; stop at paragraph text
    lo, hi = w[0] - 8, w[2] + 8
    above = [i for i, v in enumerate(words)
             if v["bbox"][3] <= w[1] + 1 and w[1] - v["bbox"][3] <= 600
             and min(v["bbox"][2], hi) - max(v["bbox"][0], lo) > 0]
    above.sort(key=lambda i: -words[i]["bbox"][3])
    hit_lines = []
    for i in above:
        li = line_of[i]
        if li in para:
            break
        if not _is_header(words[i]["text"]):
            continue
        if li not in hit_lines:
            if len(hit_lines) == 3:
                break
            hit_lines.append(li)
        for j in _phrase(words, lines[li], i):
            if j not in col:
                col.append(j)
    # caption scale: nearest line above (<= 900 px, paragraphs allowed) naming a scale
    best, best_d = None, 1e9
    for li, ln in enumerate(lines):
        if not any(SCALE_RE.search(words[i]["text"]) for i in ln):
            continue
        d = w[1] - max(words[i]["bbox"][3] for i in ln)
        if -2 <= d <= 900 and d < best_d:
            best, best_d = li, d
    if best is not None:
        ln = lines[best]
        hit = [k for k, i in enumerate(ln) if SCALE_RE.search(words[i]["text"])]
        unit += [ln[j] for k in hit for j in range(max(0, k - 2), min(len(ln), k + 2))
                 if ln[j] not in unit]
    return dict(row=row, col=col, unit=unit)


def closure_words(words, seed_ids, page_w):
    """Seeds first, then row labels, column headers, units (priority order)."""
    seen = list(dict.fromkeys(seed_ids))
    rels = [relations(words, s, page_w) for s in seed_ids if is_num(words[s]["text"])]
    for key in ("row", "col", "unit"):
        for r in rels:
            for i in r[key]:
                if i not in seen:
                    seen.append(i)
    return seen


# ---------------------------------------------------------------- token cells
class Grid:
    """Maps OCR-pixel boxes to merged visual-token cells (row-major index)."""

    def __init__(self, orig_wh, grid_hw, patch=28):
        self.W, self.H = orig_wh
        self.gh, self.gw = grid_hw
        self.cw = self.W / self.gw
        self.ch = self.H / self.gh
        self.n = self.gh * self.gw

    def cells(self, bbox):
        x0, y0, x1, y1 = bbox
        c0 = int(np.clip((x0 + 1) // self.cw, 0, self.gw - 1))
        c1 = int(np.clip((x1 - 1) // self.cw, 0, self.gw - 1))
        r0 = int(np.clip((y0 + 1) // self.ch, 0, self.gh - 1))
        r1 = int(np.clip((y1 - 1) // self.ch, 0, self.gh - 1))
        return [r * self.gw + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def word_cells(self, words, ids):
        out = []
        for i in ids:
            for c in self.cells(words[i]["bbox"]):
                if c not in out:
                    out.append(c)
        return out

    def ring(self, seed_cells, m):
        """m cells nearest (Chebyshev) to the seed cells, excluding them."""
        if m <= 0 or not seed_cells:
            return []
        rr, cc = np.divmod(np.arange(self.n), self.gw)
        sr, sc = np.divmod(np.array(seed_cells), self.gw)
        dist = np.min(np.maximum(np.abs(rr[:, None] - sr[None]), np.abs(cc[:, None] - sc[None])), 1)
        dist[np.array(seed_cells)] = 10**9
        order = np.lexsort((np.arange(self.n), dist))
        return [int(c) for c in order[:m]]


def fill(base, scores, K):
    """Truncate/extend an ordered cell list to exactly K using attention scores."""
    keep = list(dict.fromkeys(base))[:K]
    if len(keep) < K:
        s = set(keep)
        for c in np.argsort(-scores):
            if len(keep) >= K:
                break
            if int(c) not in s:
                keep.append(int(c))
                s.add(int(c))
    return keep


def attn_closure(words, grid, scores, K, page_w, clos_fn=None):
    """Greedy: visit words by max-cell attention; numeric words pull their closure.

    clos_fn(words, seed_ids) -> ordered word ids (seeds first); defaults to the heuristic.
    """
    if clos_fn is None:
        clos_fn = lambda ws, ids: closure_words(ws, ids, page_w)  # noqa: E731
    wscore = []
    for i, w in enumerate(words):
        cs = grid.cells(w["bbox"])
        wscore.append(max(scores[c] for c in cs))
    keep, s = [], set()
    for i in np.argsort(-np.array(wscore)):
        if len(keep) >= K:
            break
        ids = clos_fn(words, [int(i)]) if is_num(words[i]["text"]) else [int(i)]
        for c in grid.word_cells(words, ids):
            if c not in s and len(keep) < K:
                keep.append(c)
                s.add(c)
    return fill(keep, scores, K)


def group_closure(words, grid, scores, K, page_w, clos_fn=None, max_groups=None):
    """Query-conditioned group selection: score each numeric word's closure *as a group*
    (mean over its words of max-cell attention), add whole groups best-first, fill by attention.

    Unlike attn_closure, a distractor value whose row/column labels the question does not
    mention gets a low group score, so it is not promoted into "meaningful" evidence.
    """
    if clos_fn is None:
        clos_fn = lambda ws, ids: closure_words(ws, ids, page_w)  # noqa: E731
    wmax = np.array([max(scores[c] for c in grid.cells(w["bbox"])) for w in words])
    groups = []
    for i, w in enumerate(words):
        if not is_num(w["text"]):
            continue
        g = clos_fn(words, [i])
        groups.append((float(wmax[g].mean()), g))
    groups.sort(key=lambda t: -t[0])
    keep, s = [], set()
    for _, g in groups[:max_groups]:
        if len(keep) >= K:
            break
        for c in grid.word_cells(words, g):
            if c not in s and len(keep) < K:
                keep.append(c)
                s.add(c)
    return fill(keep, scores, K)
