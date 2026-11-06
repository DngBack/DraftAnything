"""Reports for E3 (test confirmation) and E4 (answerability-scored groups)."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_e1 import acc, boot, fmt, groups

ROOT = Path(__file__).resolve().parents[1]

E5 = [("top4_learn", "B_attn@0.05"), ("top8_learn", "B_attn@0.1"), ("full", "top8_learn")] + \
    [(a.format(r=r), b.format(r=r)) for r in (0.02, 0.05, 0.1, 0.25) for a, b in
     [("G_learn@{r}", "B_attn@{r}"), ("C_gold@{r}", "G_learn@{r}"), ("full", "G_learn@{r}")]]
E6 = [(a.format(r=r), b.format(r=r)) for r in (0.02, 0.05, 0.1) for a, b in
      [("H25@{r}", "B_attn@{r}"), ("H50@{r}", "B_attn@{r}"), ("H50@{r}", "G_learn@{r}"), ("G_learn@{r}", "B_attn@{r}")]]
FINAL = [("closure", "values"), ("closure", "local"), ("full", "closure")] + \
    [(a.format(r=r), b.format(r=r)) for r in (0.02, 0.05, 0.1, 0.25) for a, b in
     [("H50@{r}", "B_attn@{r}"), ("H25@{r}", "B_attn@{r}"), ("G_learn@{r}", "B_attn@{r}"), ("H50@{r}", "G_learn@{r}"),
      ("C_attn@{r}", "B_attn@{r}"), ("G_attn@{r}", "B_attn@{r}"), ("C_gold@{r}", "L_gold@{r}"), ("C_gold@{r}", "B_gold@{r}"),
      ("C_gold@{r}", "H50@{r}"), ("full", "H50@{r}")]]
PAIRS = {
    "final": FINAL,
    "e6": E6, "e6test": E6,
    "e5": E5, "e5test": E5,
    "e3": [("closure", "values"), ("closure", "local"), ("full", "closure")]
    + [(a.format(r=r), b.format(r=r)) for r in (0.02, 0.1, 0.25) for a, b in
       [("C_gold@{r}", "L_gold@{r}"), ("C_gold@{r}", "B_gold@{r}"), ("C_attn@{r}", "B_attn@{r}"),
        ("G_attn@{r}", "B_attn@{r}"), ("G_attn@{r}", "C_attn@{r}"), ("full", "C_attn@{r}")]],
    "e4": [("top_lex", "top_vlm"), ("top_mix", "top_lex")]
    + [(a.format(r=r), b.format(r=r)) for r in (0.02, 0.05, 0.1) for a, b in
       [("G_lex@{r}", "B_attn@{r}"), ("G_vlm@{r}", "B_attn@{r}"), ("G_mix@{r}", "B_attn@{r}"),
        ("G_lex@{r}", "G_vlm@{r}"), ("C_gold@{r}", "G_lex@{r}"), ("C_gold@{r}", "G_mix@{r}")]],
}


def main(prefix, tag):
    recs = [json.loads(l) for l in open(ROOT / "results" / f"{prefix}_{tag}.jsonl")]
    conds = list(recs[0]["res"])
    G = groups(recs)
    L = [f"# {prefix.upper()} — {tag}", "", f"n = {len(recs)} câu, {len({r['uid'] for r in recs})} tài liệu.", ""]
    if prefix in ("e4", "e5", "e5test", "e6", "e6test", "final"):
        L += ["## Xếp hạng nhóm: recall@k của nhóm gold (giá trị đúng)", "",
              "| scorer | @1 | @2 | @4 | @8 |", "|---|---|---|---|---|"]
        for k in recs[0]["recall_at"]:
            L.append(f"| {k} | " + " | ".join(f"{100 * np.mean([r['recall_at'][k][m] for r in recs]):.1f}"
                                              for m in ("1", "2", "4", "8")) + " |")
        L += ["", f"Số nhóm ứng viên trung bình mỗi trang: {np.mean([r['n_groups'] for r in recs]):.0f}; "
              f"thời gian chấm VLM: {np.mean([r['t_vlm_score'] for r in recs]):.1f} s/câu.", ""]
    L += ["## Độ chính xác", "", "| Điều kiện | K | value acc | scale acc | fact recall |", "|---|---|---|---|---|"]
    for c in conds:
        L.append(f"| `{c}` | {np.mean([r['res'][c]['K'] for r in recs]):.0f} | {100 * acc(recs, c)[0]:.1f} | "
                 f"{100 * acc(recs, c, 's')[0]:.1f} | {100 * np.mean([r['res'][c]['fact_rec'] for r in recs]):.0f}% |")
    for key, nm in (("v", "value acc"), ("s", "scale acc")):
        L += ["", f"## Hiệu cặp — {nm} (điểm %, CI 95% bootstrap theo tài liệu; * = CI không chứa 0)", "",
              "| Cặp | " + " | ".join(f"{g} (n={len(G[g])})" for g in G) + " |", "|---|" + "---|" * len(G)]
        for a, b in PAIRS[prefix]:
            if a in conds and b in conds:
                L.append(f"| `{a}` − `{b}` | " + " | ".join(fmt(boot(G[g], a, b, key)) for g in G) + " |")
    (ROOT / "results" / f"{prefix}_{tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", choices=["e3", "e4", "e5", "e5test", "e6", "e6test", "final"])
    ap.add_argument("--tag", default="qwen25vl7b")
    a = ap.parse_args()
    main(a.prefix, a.tag)
