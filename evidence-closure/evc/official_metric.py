"""Exact-match rule ported from the official TAT-QA evaluator (NExTplusplus/TAT-QA: tatqa_metric.py, tatqa_utils.py).

Behaviour kept as in the original, including rounding to 2 decimals before applying the scale and the numeric
coupling of value and scale (5 thousand == 5000 with empty scale). Only EM is ported (no F1).
"""
import re
import string

EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"
EXCLUDE = set(string.punctuation)


def scale_to_num(scale):
    scale = scale.lower()
    if "hundred" in scale:
        return 100
    if "thousand" in scale:
        return 1000
    if "million" in scale:
        return 1000000
    if "billion" in scale:
        return 1000000000
    if "percent" in scale:
        return 0.01
    return 1


def _clean_num(text):
    return "".join(ch for ch in str(text) if ch not in EXCLUDE_IN_NUM)


def is_number(text):
    try:
        words = " ".join(_clean_num(w) for w in text.split()).split()
        if not words:
            return False
        num = float(words[0])
        if num != num:
            return False
        return not (len(words) >= 2 and scale_to_num(words[1]) == 1)
    except ValueError:
        return False


def _one_num(s):
    groups = re.findall(r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)", _clean_num(s))
    if not groups or groups[0][0] == "":
        return None
    n = groups[0][0]
    return float(n) if "." in n else int(n)


def _word_scale(x):
    for m in re.finditer(r"([\d.]+\s?[a-zA-Z]+)", x):
        return scale_to_num(m.group(0).lower())
    return 1


def to_number(text):
    num = _one_num(text)
    if num is None:
        return None
    neg = -1 if re.findall(r"(\([\d.\s]+\))", text.strip()) else 1
    pct = 0.01 if re.findall(r"([\d.\s]+%)", text.strip()) else 1
    return round(num * _word_scale(text) * neg * pct, 4)


def _norm_token(tok):
    t = tok.lower()
    if not is_number(t):
        t = "".join(c for c in t if c not in EXCLUDE)
    if is_number(t):
        t = str(to_number(t))
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


def normalize_answer(text):
    return " ".join(p for p in (_norm_token(t) for t in text.split(" ")) if p.strip()).strip()


def answer_str(answers, scale):
    out = []
    for a in sorted(answers):
        s = str(a)
        if is_number(s):
            n = to_number(s)
            if n is None:
                if scale:
                    s = s + " " + str(scale)
            elif "%" in s:
                s = "%.4f" % n
            else:
                s = "%.4f" % (round(n, 2) * scale_to_num(scale))
        elif scale:
            s = s + " " + str(scale)
        out.append(s)
    return " ".join(out)


def exact_match(pred, pred_scale, r):
    """pred: list of answer strings. r: dict with answer_type, answer, scale."""
    if not pred:
        return 0.0
    t, a = r["answer_type"], r["answer"]
    if t in ("span", "multi-span"):
        gold = [str(x) for x in a]
    elif t == "count":
        gold = [str(int(a))]
    else:
        gold = [str(a)]
    g = normalize_answer(answer_str(gold, r["scale"]))
    cands = [answer_str(pred, pred_scale)]
    if len(pred) == 1 and not pred_scale and "%" not in pred[0] and is_number(pred[0]):  # add_percent_pred
        n = to_number(pred[0])
        if n is not None:
            cands.append("%.4f" % n)
    return float(any(normalize_answer(c) == g for c in cands))
