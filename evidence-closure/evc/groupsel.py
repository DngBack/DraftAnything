"""Learned query-conditioned group selector (OCR + layout only, no VLM at selection time).

A group = numeric seed word + its heuristic closure (row label, column header, unit).
Supervision: TAT-DQA gold facts (a group is positive iff its seed is a gold fact word).
Model: small MLP scorer trained with a listwise (multi-positive softmax) loss per question.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import torch
import torch.nn as nn

from .closure import YEAR_RE, closure_words, layout, relations
from .data import is_num
from .learned import norm, q_tokens

PCT_Q = re.compile(r"percent|percentage|%|proportion|ratio|rate", re.I)
CHANGE_Q = re.compile(r"change|increase|decrease|difference|growth", re.I)


def page_groups(words, page_w):
    """All candidate groups on a page: dict(seed, words, row, col, unit)."""
    out = []
    for i, w in enumerate(words):
        if not is_num(w["text"]):
            continue
        r = relations(words, i, page_w)
        out.append(dict(seed=i, words=closure_words(words, [i], page_w), row=r["row"], col=r["col"], unit=r["unit"]))
    return out


def _toks(words, ids):
    return {norm(words[j]["text"]) for j in ids} - {""}


def group_features(words, groups, question, page_w, page_h):
    qt = q_tokens(question)
    qyears = {t for t in qt if YEAR_RE.match(t)}
    ql = question.lower()
    lines, line_of, para = layout(words, page_w)
    F = []
    for g in groups:
        s = words[g["seed"]]
        rt, ct, ut = _toks(words, g["row"]), _toks(words, g["col"]), _toks(words, g["unit"])
        rstr = " ".join(words[j]["text"] for j in g["row"]).lower()
        cyears = {t for t in ct if YEAR_RE.match(t)}
        t = s["text"]
        try:
            mag = np.log10(abs(float(re.sub(r"[^\d.]", "", t) or 0)) + 1)
        except ValueError:
            mag = 0.0
        F.append([
            len(rt & qt), len(rt & qt) / max(1, len(rt)), len(rt & qt) / max(1, len(qt)),
            len(ct & qt), len(ct & qt) / max(1, len(ct)),
            float(bool(qyears & cyears)), float(bool(qyears) and bool(cyears) and not (qyears & cyears)),
            float(bool(qyears)), len(cyears) / 3.0,
            SequenceMatcher(None, rstr, ql).find_longest_match(0, len(rstr), 0, len(ql)).size / max(1, len(rstr)),
            float(len(g["row"]) == 0), float(len(g["col"]) == 0), float(len(g["unit"]) > 0),
            float("%" in t), float(bool(PCT_Q.search(question))), float("%" in t and bool(PCT_Q.search(question))),
            float(bool(CHANGE_Q.search(question))), float("(" in t), mag / 10.0,
            s["bbox"][0] / page_w, s["bbox"][1] / page_h, float(line_of[g["seed"]] in para),
            float(bool(YEAR_RE.match(t))), len(groups) / 200.0,
        ])
    return np.asarray(F, dtype=np.float32)


class GroupScorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def train_listwise(samples, epochs=15, seed=0):
    """samples: list of (X [n_groups, d], y [n_groups] in {0,1}) with at least one positive."""
    torch.manual_seed(seed)
    d = samples[0][0].shape[1]
    m = GroupScorer(d)
    allX = torch.tensor(np.concatenate([x for x, _ in samples]))
    m.mu.copy_(allX.mean(0))
    m.sd.copy_(allX.std(0) + 1e-4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    data = [(torch.tensor(x), torch.tensor(y)) for x, y in samples]
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        tot = 0.0
        for k in rng.permutation(len(data)):
            x, y = data[k]
            lp = torch.log_softmax(m(x), 0)
            loss = -(lp * y).sum() / y.sum()  # multi-positive listwise
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"  epoch {ep} loss {tot / len(data):.4f}", flush=True)
    return m.eval()
