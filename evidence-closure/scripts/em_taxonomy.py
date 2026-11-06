"""Error anatomy under official EM, on TRAIN only (out-of-fold v2-em selections)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import h3_policy as H  # noqa: E402
from evc.metrics import parse_nums, value_correct  # noqa: E402
from h3c_policy import val_em  # noqa: E402
from official_rescore import em, load, pred_of, sel_v2  # noqa: E402
from run_e8 import read_recall, safe_eval  # noqa: E402


def gold_num(r):
    a = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
    try:
        return float(str(a).replace(",", "").replace("$", "").replace("%", ""))
    except ValueError:
        return None


def pred_num(v, r):
    t = r[v] if v != "direct" and r[v] is not None else r["direct"]
    if v != "direct" and H.has_expr(t):
        return safe_eval(t)
    n = re.findall(r"-?\(?\$?\d[\d,]*\.?\d*\)?", r["direct"])
    if not n:
        return None
    s = n[0]
    x = float(re.sub(r"[()$,]", "", s))
    return -abs(x) if s.startswith("(") else x


def kind(v, s, r):
    if em(v, s, r):
        return "ok"
    if em(v, r["scale"], r):
        return "scale_only"
    t = r["answer_type"]
    g, p = gold_num(r), pred_num(v, r)
    if t == "arithmetic" and g is not None and p is not None:
        if abs(abs(p) - abs(g)) < 0.01 and (p < 0) != (g < 0):
            return "sign"
        if abs(p * 100 - g) < 0.01 or abs(abs(p) * 100 - abs(g)) < 0.01:
            return "pct_x100"
        if abs(abs(p) - abs(g)) <= 0.005 * max(abs(g), 1e-9):
            return "rounding"
    if t in ("span", "multi-span"):
        if value_correct(r["direct"], r["answer"]):
            return "span_format"
        return "span_wrong"
    if t == "count":
        return "count"
    rd = read_recall(r["program"], r["facts"]) >= 1
    return "op_or_cond" if rd else "read"


for tag in ("qwen25vl7b", "qwen3vl8b"):
    R = load(tag, "train", "train")
    _, fo = H.folds(R, seed=1)
    C, ex = Counter(), {}
    for k in range(5):
        tr = [r for r, f in zip(R, fo) if f != k]
        va = [r for r, f in zip(R, fo) if f == k]
        for (v, s), r in zip(sel_v2(tr, va, 2, valf=val_em), va):
            c = kind(v, s, r)
            C[c] += 1
            if c != "ok":
                ex.setdefault(c, []).append((r["question"][:70], r["answer"], r["scale"], pred_of(v, r, s), s, r["derivation"][:40]))
    print(tag, {k: f"{v} ({100 * v / len(R):.1f}%)" for k, v in C.most_common()})
    for c, L in ex.items():
        print(" --", c)
        for e in L[:5]:
            print("    ", e)
