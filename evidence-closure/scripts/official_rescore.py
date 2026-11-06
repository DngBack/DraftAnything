"""Post-hoc sensitivity analysis: rescore the already locked H3/H3b selections with the official TAT-QA EM rule.

Selections are NOT refit on this metric: models, lambda and tables are exactly as in the locked runs (lenient labels).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import h3_policy as H  # noqa: E402
from evc.official_metric import exact_match, is_number  # noqa: E402
from h3_policy import LR, base_feats, boot, fit_models, has_expr, label, safe_eval, scale_of, val_of  # noqa: E402
from h3b_policy import ScaleClf, cell  # noqa: E402

LAM = {("qwen25vl7b", 2): 10.0, ("qwen25vl7b", 3): 0.06, ("qwen3vl8b", 2): 10.0, ("qwen3vl8b", 3): 0.0}  # from locked runs
SW = re.compile(r"\b(thousands?|millions?|billions?|percent(age)?)\b\.?", re.I)
UNIT_ONLY = re.compile(r"^\W*(in\s+)?(thousands?|millions?|billions?|percent(age)?|\$?'?000s?|[$€£]\s?(m|bn|k)|usd|dollars?|%)\W*$", re.I)


def load(tag, name, split):
    rows = [json.loads(l) for l in open(ROOT / "results" / f"h3_{name}_{tag}.jsonl")]
    for x in rows:
        x["_split"] = split
    return rows


def direct_pred(t, s):
    """Answer strings from free text; uses no gold. A scale word lives either in the string or in the label s, never both."""
    raw = [x for x in re.split(r";|\n", t) if x.strip()]
    kept = [x for x in raw if not UNIT_ONLY.match(x)] or raw
    parts = []
    for x in kept:
        x = (x.rsplit("=", 1)[1] if "=" in x else x).strip().rstrip(".")  # "a - b = c" -> the stated result
        if s:
            x = SW.sub("", x).strip().rstrip(".")
        if x and x not in parts:
            parts.append(x)
    if len(parts) == 1:
        sp = [y.strip() for y in re.split(r",\s+", parts[0])]
        if len(sp) > 1 and all(is_number(y) for y in sp):
            parts = sp
    return parts


def pred_of(v, r, s):
    if v != "direct":
        t = r[v] if r[v] is not None else r["program"]
        if has_expr(t):
            return [str(float(safe_eval(t)))]
    return direct_pred(r["direct"], s)


def em(v, s, r):
    return exact_match(pred_of(v, r, s), s, r)


def sel_v2(Rtr, R, B, valf=val_of):
    Xtr, X = np.array([base_feats(r) for r in Rtr]), np.array([base_feats(r) for r in R])
    pv = {v: LR().fit(Xtr, np.array([float(valf(v, r)) for r in Rtr])).predict(X) for v in ("direct", "program")}
    sc = ScaleClf().fit(Rtr, B >= 3).predict(R)
    return [(max(pv, key=lambda v: pv[v][i]), sc[i]) for i in range(len(R))]


def sel_rule(Rtr, R, B, min_n=5, label=label):
    S = ("direct", "ocr", "vlm") if B >= 3 else ("direct", "ocr")
    P = [(v, s) for v in ("direct", "program") for s in S]
    glob = max(P, key=lambda p: np.mean([label(r, *p) for r in Rtr]))
    by = defaultdict(list)
    for r in Rtr:
        by[cell(r)].append(r)
    ch = {k: max(P, key=lambda p: np.mean([label(r, *p) for r in L])) for k, L in by.items() if len(L) >= min_n}
    return [(lambda p: (p[0], scale_of(p[1], r)))(ch.get(cell(r), glob)) for r in R]


def sel_h3(M, R, lam):
    rec, orig = [], H.label
    def spy(r, v, s):
        rec.append((v, scale_of(s, r)))
        return orig(r, v, s)
    H.label = spy
    try:
        H.learned_policy(R, M, lam)
    finally:
        H.label = orig
    return rec


def main():
    # sanity: the gold answer must score 1.0
    L = ["# Chấm lại bằng luật EM của evaluator TAT-QA chính thức (phân tích sau, không đổi policy đã khóa)", ""]
    for tag in ("qwen25vl7b", "qwen3vl8b"):
        Rtr, Rt1, Rt2 = load(tag, "train", "train"), load(tag, "test", "test"), load(tag, "test2", "test")
        for name, R in (("test2", Rt2), ("test + test2", Rt1 + Rt2)):
            gold_ok = np.mean([exact_match([str(x) for x in r["answer"]] if isinstance(r["answer"], list) else [str(r["answer"])],
                                           r["scale"], r) for r in R])
            L += [f"## {tag}, {name}, n = {len(R)}; kiểm tra: gold tự chấm = {100 * gold_ok:.1f} %", "",
                  "| policy | B | thước đo cũ (dễ dãi) | **EM chính thức** |", "|---|---|---|---|"]
            docs = [r["uid"] for r in R]
            M = fit_models(Rtr)
            rows = {}
            def add(n, B, sel, old):
                c = np.array([em(v, s, r) for (v, s), r in zip(sel, R)])
                rows[(n, B)] = c
                L.append(f"| {n} | {B} | {100 * np.mean(old):.1f} | {100 * c.mean():.1f} |")
            add("direct", 1, [("direct", r["scale_direct"]) for r in R], [label(r, "direct", "direct") for r in R])
            add("fixed2", 2, [("program", r["scale_direct"]) for r in R], [label(r, "program", "direct") for r in R])
            for B in (2, 3):
                s = sel_rule(Rtr, R, B)
                add("rule_table", B, s, [val_of(v, r) and sc == r["scale"] for (v, sc), r in zip(s, R)])
                s = sel_h3(M, R, LAM[(tag, B)])
                add("H3-learned", B, s, [val_of(v, r) and sc == r["scale"] for (v, sc), r in zip(s, R)])
                s = sel_v2(Rtr, R, B)
                add("v2", B, s, [val_of(v, r) and sc == r["scale"] for (v, sc), r in zip(s, R)])
            L.append("")
            for B in (2, 3):
                for ref in (("fixed2", 2), ("rule_table", B), ("H3-learned", B)):
                    e, lo, hi = boot(rows[("v2", B)], rows[ref], docs)
                    L.append(f"- B = {B}: v2 − {ref[0]} = {100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" + (" *" if lo > 0 or hi < 0 else ""))
            L.append("")
            for t in ("arithmetic", "span", "multi-span", "count"):
                ii = [i for i, r in enumerate(R) if r["answer_type"] == t]
                L.append(f"- {t} (n={len(ii)}): fixed2 {100 * rows[('fixed2', 2)][ii].mean():.1f}, v2 B2 {100 * rows[('v2', 2)][ii].mean():.1f}, v2 B3 {100 * rows[('v2', 3)][ii].mean():.1f}")
            L.append("")
    (ROOT / "results" / "official_rescore.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
