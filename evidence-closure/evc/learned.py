"""Learned relational closure: pairwise (seed value word, candidate word) relevance from layout.

Weak supervision (TAT-DQA train): for each numeric fact word, a candidate is positive if its
normalised token occurs in the question (row/column label the question names) or it names the
gold scale. Two models:
  - geo   : layout features only (question NOT used at inference) -> drop-in for the heuristic
  - query : geo + "token occurs in question" (query-conditioned group selector)
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn

from .closure import ALPHA_RE, SCALE_RE, YEAR_RE, _vov, layout
from .data import is_num

STOP = set("the a an of in on for to and or is was were what which how much many by from as at with "
           "that this be are its it their there than between respective respectively value values "
           "total change average".split())
CUR = {"$", "€", "£", "rmb", "us$"}


def norm(t):
    return re.sub(r"[^a-z0-9.]", "", t.lower()).strip(".")


def q_tokens(q):
    return {norm(t) for t in q.split()} - STOP - {""}


def candidates(words, s, page_w, page_h, max_dy=900):
    """Candidate ids for seed s: same line, or above within max_dy px."""
    w = words[s]["bbox"]
    out = []
    for i, v in enumerate(words):
        if i == s:
            continue
        b = v["bbox"]
        if _vov(b, w) >= 0.5 or (b[3] <= w[1] + 2 and w[1] - b[3] <= max_dy):
            out.append(i)
    return out


def features(words, s, cands, page_w, page_h):
    lines, line_of, para = layout(words, page_w)
    w = words[s]["bbox"]
    sw = max(w[2] - w[0], 1.0)
    sl = line_of[s]
    F = []
    # rank of header-like x-overlapping lines above the seed (nearest = 1)
    above_lines = sorted({line_of[i] for i in cands
                          if words[i]["bbox"][3] <= w[1] + 2
                          and min(words[i]["bbox"][2], w[2] + 8) - max(words[i]["bbox"][0], w[0] - 8) > 0
                          and (ALPHA_RE.search(words[i]["text"]) or YEAR_RE.match(words[i]["text"]))},
                         key=lambda li: -max(words[j]["bbox"][3] for j in lines[li]))
    rank = {li: k + 1 for k, li in enumerate(above_lines)}
    for i in cands:
        v = words[i]
        b = v["bbox"]
        t = v["text"]
        li = line_of[i]
        same = _vov(b, w) >= 0.5
        xo = max(0.0, min(b[2], w[2]) - max(b[0], w[0])) / sw
        xo8 = float(min(b[2], w[2] + 8) - max(b[0], w[0] - 8) > 0)
        between = 0
        if same:
            lo, hi = sorted((b[2], w[0])) if b[0] < w[0] else sorted((w[2], b[0]))
            between = sum(1 for j in lines[sl] if lo <= words[j]["bbox"][0] <= hi and is_num(words[j]["text"]))
        ln = lines[li]
        F.append([
            (b[0] + b[2] - w[0] - w[2]) / 2 / page_w,            # dx centre
            (w[1] - b[3]) / page_h,                               # dy (>0 above)
            float(same), float(same and b[2] <= w[0] + 2), float(same and b[0] >= w[2] - 2),
            min(xo, 2.0), xo8,
            float(is_num(t)), float(bool(YEAR_RE.match(t))), float(bool(ALPHA_RE.search(t))),
            float(bool(SCALE_RE.search(t))), float(norm(t) in CUR or "%" in t),
            float(li in para), len(ln) / 20.0, float(ln[0] == i), b[0] / page_w,
            min(between, 10) / 10.0, min(rank.get(li, 0), 6) / 6.0, float(rank.get(li, 0) == 1),
            float(li == sl), float(norm(t) in STOP),
        ])
    return np.asarray(F, dtype=np.float32)


N_GEO = 21


def labels(words, cands, question, scale):
    qt = q_tokens(question)
    y = []
    for i in cands:
        n = norm(words[i]["text"])
        pos = n in qt and len(n) >= 2
        if scale and scale != "percent" and scale[:5] in n:
            pos = True
        if scale == "percent" and "%" in words[i]["text"]:
            pos = True
        y.append(float(pos))
    return np.asarray(y, dtype=np.float32)


def q_feature(words, cands, question):
    qt = q_tokens(question)
    return np.asarray([[float(norm(words[i]["text"]) in qt)] for i in cands], dtype=np.float32)


class PairMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def build_pairs(items, words_of, use_query):
    X, Y, G = [], [], []
    for gi, it in enumerate(items):
        words, (W, H) = words_of(it)
        for s in it["fact_ids"]:
            if not is_num(words[s]["text"]):
                continue
            c = candidates(words, s, W, H)
            if not c:
                continue
            f = features(words, s, c, W, H)
            if use_query:
                f = np.concatenate([f, q_feature(words, c, it["question"])], 1)
            X.append(f)
            Y.append(labels(words, c, it["question"], it["scale"]))
            G.append(np.full(len(c), gi))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def train(X, Y, epochs=8, seed=0, device="cpu"):
    torch.manual_seed(seed)
    m = PairMLP(X.shape[1]).to(device)
    Xt, Yt = torch.tensor(X, device=device), torch.tensor(Y, device=device)
    m.mu.copy_(Xt.mean(0))
    m.sd.copy_(Xt.std(0) + 1e-4)
    pw = torch.tensor((1 - Y.mean()) / max(Y.mean(), 1e-4), device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for k in range(0, n, 4096):
            idx = perm[k:k + 4096]
            loss = lossf(m(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"  epoch {ep} loss {tot / n:.4f}", flush=True)
    return m.eval()


class LearnedClosure:
    """closure_words-compatible: seeds first, then candidates with p >= tau (max k_max per seed)."""

    def __init__(self, model, tau, use_query=False, k_max=12):
        self.m, self.tau, self.q, self.k_max = model, tau, use_query, k_max

    @torch.no_grad()
    def scores(self, words, s, page_w, page_h, question=None):
        c = candidates(words, s, page_w, page_h)
        if not c:
            return [], np.zeros(0)
        f = features(words, s, c, page_w, page_h)
        if self.q:
            f = np.concatenate([f, q_feature(words, c, question)], 1)
        p = torch.sigmoid(self.m(torch.tensor(f))).numpy()
        return c, p

    def __call__(self, words, seed_ids, page_w, page_h, question=None):
        out = list(dict.fromkeys(seed_ids))
        per_seed = []
        for s in seed_ids:
            if not is_num(words[s]["text"]):
                continue
            c, p = self.scores(words, s, page_w, page_h, question)
            o = np.argsort(-p)
            per_seed.append([c[j] for j in o[: self.k_max] if p[j] >= self.tau])
        # interleave seeds so every value gets its best relations before anyone's tail
        for r in range(self.k_max):
            for lst in per_seed:
                if r < len(lst) and lst[r] not in out:
                    out.append(lst[r])
        return out
