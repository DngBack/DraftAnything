"""H3b (docs/protocol_h3b.md): stronger baselines + factorized learned selector (value source x scale classifier).

usage: h3b_policy.py --tag qwen25vl7b          # 5-fold CV on train only (development)
       h3b_policy.py --tag qwen25vl7b --test   # fit on all train, evaluate once on test2
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.metrics import parse_nums  # noqa: E402
from h3_policy import (LR, QRE, base_feats, boot, evaluate, failure_type, folds, has_expr, label, num,  # noqa: E402
                       scale_of, val_of)

CLASSES = ("", "thousand", "million", "billion", "percent")
YEAR = re.compile(r"^\D*(19|20)\d{2}\D*$")
QX = {k: re.compile(p, re.I) for k, p in dict(
    howmany=r"how many", which=r"^(which|what) (year|month|segment|company|type|item|was the name)", year=r"\byear\b",
    inthou=r"thousand", inmil=r"million", ratio=r"\bratio\b|times").items()}


def onehot(x):
    return [float(x == c) for c in CLASSES]


def scale_feats(r, with_vlm):
    d = r["direct"]
    v = num(r["program"]) if has_expr(r["program"]) else num(d)
    nums = parse_nums(d)
    f = base_feats(r) + onehot(r["scale_direct"]) + onehot(r["scale_ocr"])
    f += [float(bool(nums)), float(bool(YEAR.match(d.strip()))), sum(c.isalpha() for c in d) / max(1, len(d)),
          len(nums) / 3, math.log10(abs(v) + 1) / 4 if v is not None else -1, float(v is not None and abs(v) < 1),
          float("$" in d)]
    f += [float(bool(rx.search(r["question"]))) for rx in QX.values()]
    if with_vlm:
        f += onehot(r["scale_vlm"])
    return f


class ScaleClf:
    """One-vs-rest logistic regression over the 5 TAT-DQA scales; predicts the gold scale directly."""

    def fit(self, R, with_vlm):
        self.w = with_vlm
        X = np.array([scale_feats(r, with_vlm) for r in R])
        self.m = {c: LR(l2=3.0).fit(X, np.array([float(r["scale"] == c) for r in R])) for c in CLASSES
                  if sum(r["scale"] == c for r in R) >= 3}
        return self

    def predict(self, R):
        X = np.array([scale_feats(r, self.w) for r in R])
        P = {c: m.predict(X) for c, m in self.m.items()}
        return [max(P, key=lambda c: P[c][i]) for i in range(len(R))]


def v2_policy(Rtr, R, B):
    """Value source in {direct, program} by P(value correct); scale by classifier. B=3 adds scale_vlm (1 call)."""
    with_vlm = B >= 3
    Xtr = np.array([base_feats(r) for r in Rtr])
    X = np.array([base_feats(r) for r in R])
    pv = {v: LR().fit(Xtr, np.array([float(val_of(v, r)) for r in Rtr])).predict(X) for v in ("direct", "program")}
    sc = ScaleClf().fit(Rtr, with_vlm).predict(R)
    out = [float(val_of(max(pv, key=lambda v: pv[v][i]), r) and sc[i] == r["scale"]) for i, r in enumerate(R)]
    return np.array(out), np.full(len(R), 3 if with_vlm else 2)


def cell(r):
    d = r["direct"]
    return (has_expr(r["program"]), bool(parse_nums(d)), bool(YEAR.match(d.strip())), r["scale_direct"], r["scale_ocr"],
            bool(QRE["pct"].search(r["question"])))


def rule_table(Rtr, R, B, min_n=5):
    """Lookup-table baseline: for each feature cell pick the (value, scale) source pair best on train."""
    S = ("direct", "ocr", "vlm") if B >= 3 else ("direct", "ocr")
    P = [(v, s) for v in ("direct", "program") for s in S]
    glob = max(P, key=lambda p: np.mean([label(r, *p) for r in Rtr]))
    by = defaultdict(list)
    for r in Rtr:
        by[cell(r)].append(r)
    choice = {k: max(P, key=lambda p: np.mean([label(r, *p) for r in L])) for k, L in by.items() if len(L) >= min_n}
    out = [label(r, *choice.get(cell(r), glob)) for r in R]
    return np.array(out), np.full(len(R), 3 if B >= 3 else 2)


def oracle(R, B):
    S = ("direct", "ocr", "vlm") if B >= 3 else ("direct", "ocr")
    return np.array([float(any(label(r, v, s) for v in ("direct", "program") for s in S)) for r in R])


def run(Rtr, Rte, B):
    res, par = evaluate(Rtr, Rte, B)  # H3 policies, unchanged
    ren = lambda k: "H3-learned" if k == "learned" else ("fixed3" if k.startswith("fixed3_") else k)  # noqa: E731
    res = {ren(k): v for k, v in res.items()}
    res["rule_table"] = rule_table(Rtr, Rte, B)
    res["v2"] = v2_policy(Rtr, Rte, B)
    return res, par


def cv(R, B, k=5):
    """Out-of-fold evaluation on train (development only)."""
    _, fo = folds(R, seed=1)
    acc = defaultdict(list)
    for kf in range(k):
        tr = [r for r, f in zip(R, fo) if f != kf]
        va = [r for r, f in zip(R, fo) if f == kf]
        res, _ = run(tr, va, B)
        for n, (c, _) in res.items():
            acc[n].append(c)
        acc["oracle"].append(oracle(va, B))
    return {n: np.concatenate(v) for n, v in acc.items()}, [r["uid"] for kf in range(k) for r, f in zip(R, fo) if f == kf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    Rtr = [json.loads(l) for l in open(ROOT / "results" / f"h3_train_{a.tag}.jsonl")]
    L = []
    for B in (2, 3):
        if a.test:
            Rte = [json.loads(l) for l in open(ROOT / "results" / f"h3_test2_{a.tag}.jsonl")]
            res, par = run(Rtr, Rte, B)
            acc = {n: c for n, (c, _) in res.items()}
            cost = {n: k.mean() for n, (_, k) in res.items()}
            acc["oracle"] = oracle(Rte, B)
            docs = [r["uid"] for r in Rte]
            head = f"test2, n = {len(Rte)} câu, {len(set(docs))} tài liệu; tham số H3: {par}"
        else:
            acc, docs = cv(Rtr, B)
            cost = {}
            head = f"5-fold CV theo tài liệu trên train, n = {len(docs)}"
        L += [f"## B = {B}", "", head, "", "| policy | accuracy | chi phí TB |", "|---|---|---|"]
        for n, c in acc.items():
            L.append(f"| {n} | {100 * c.mean():.1f} | {cost.get(n, float('nan')):.2f} |")
        base = {n: c for n, c in acc.items() if n not in ("v2", "oracle")}
        bk = max(base, key=lambda n: base[n].mean())
        for ref in dict.fromkeys([bk, "rule_table", "H3-learned"]):
            e, lo, hi = boot(acc["v2"], acc[ref], docs)
            L.append(f"\n**v2 − {ref}** = {100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" + (" *" if lo > 0 or hi < 0 else ""))
        if a.test and B == 3:
            ft = [failure_type(r, "direct") for r in Rte]
            L += ["", "Recovery trên các câu Fixed-2 sai (số câu đúng):", "", "| loại | n | " + " | ".join(acc) + " |",
                  "|---|---|" + "---|" * len(acc)]
            for t in ("scale", "read", "reason", "span"):
                ii = [i for i, x in enumerate(ft) if x == t]
                if ii:
                    L.append(f"| {t} | {len(ii)} | " + " | ".join(str(int(acc[n][ii].sum())) for n in acc) + " |")
        L.append("")
    L = [f"# H3b — {a.tag} — {'test2' if a.test else 'train CV (dev)'}", "",
         "Metric chính thức: giá trị đúng (strict) và scale = gold. Tất cả policy chỉ học trên train.", ""] + L
    out = ROOT / "results" / f"h3b_{a.tag}_{'test2' if a.test else 'dev'}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
