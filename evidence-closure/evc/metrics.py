"""Answer scoring shared by every condition (paired comparisons)."""
from __future__ import annotations

import re
from collections import Counter

NUM_FIND = re.compile(r"\(?-?\$?\d[\d,]*\.?\d*\)?")
SCALE_WORDS = {
    "thousand": ("thousand", "thousands", "000s", "'000", "k"),
    "million": ("million", "millions", "mn", "m"),
    "billion": ("billion", "billions", "bn", "b"),
    "percent": ("percent", "%", "percentage"),
}


def parse_nums(s: str) -> list[float]:
    out = []
    for m in NUM_FIND.findall(s):
        t = m.replace("$", "").replace(",", "").strip("()")
        try:
            out.append(abs(float(t)))
        except ValueError:
            pass
    return out


def _decimals(s: str) -> int:
    t = s.replace(",", "")
    m = re.search(r"\.(\d+)", t)
    return len(m.group(1)) if m else 0


def num_match(p: float, g: float, dec: int) -> bool:
    if round(p, dec) == round(g, dec):
        return True
    return abs(p - g) <= 0.005 * max(abs(g), 1e-9)


def _toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def text_f1(p, g):
    pt, gt = _toks(p), _toks(g)
    if not pt or not gt:
        return float(pt == gt)
    c = sum((Counter(pt) & Counter(gt)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(pt), c / len(gt)
    return 2 * pr * rc / (pr + rc)


def gold_list(ans) -> list[str]:
    if isinstance(ans, list):
        return [str(a) for a in ans]
    return [str(ans)]


def value_correct(pred: str, answer) -> bool:
    golds = gold_list(answer)
    pnums = parse_nums(pred)
    for g in golds:
        gn = parse_nums(g)
        alpha = re.search(r"[A-Za-z]{3,}", g)
        if gn and not alpha:
            if not any(num_match(p, gn[0], _decimals(g)) for p in pnums):
                return False
        else:
            if text_f1(pred, g) < 0.5 and g.lower() not in pred.lower():
                return False
    return True


def scale_correct(pred: str, scale: str) -> bool | None:
    if not scale:
        return None
    toks = re.findall(r"[a-z]+|%|'000|000s", pred.lower())
    return any(t in SCALE_WORDS.get(scale, (scale,)) for t in toks if len(t) > 1 or t == "%")
