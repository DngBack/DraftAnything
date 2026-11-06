"""E2 report: learned vs heuristic closure in the VLM; joins E1 (B_attn / C_attn / B_gold) by qid."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_e1 import acc, boot, fmt, groups  # noqa: E402

RHOS = (0.02, 0.05, 0.1)


def main(tag):
    e2 = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "results" / f"e2_{tag}.jsonl")}
    e1 = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "results" / f"e1_{tag}.jsonl")}
    recs = []
    for q, r in e2.items():
        if q not in e1:
            continue
        r = dict(r)
        res = dict(r["res"])
        for k, v in e1[q]["res"].items():
            res.setdefault(k if k not in ("values", "closure") else f"e1_{k}", v)
        r["res"] = res
        recs.append(r)
    G = groups(recs)
    sizes = {k: np.mean([len(r["closure_words"][k]) for r in recs]) for k in ("heur", "geo", "query")}
    L = [f"# E2 — learned closure, {tag}", "",
         f"n = {len(recs)} câu (giao E1∩E2). Số word trung bình (gồm seed): heuristic {sizes['heur']:.1f}, "
         f"geo {sizes['geo']:.1f}, query {sizes['query']:.1f}.", "",
         "| Điều kiện | K | value acc | scale acc |", "|---|---|---|---|"]
    conds = ["full", "values", "closure", "closure_geo", "closure_query"] + \
        [f"{m}@{r}" for r in RHOS for m in ("B_attn", "C_attn", "C_attn_geo", "G_attn", "G_attn_geo", "B_gold", "C_gold", "C_gold_geo", "C_gold_query")]
    for c in conds:
        L.append(f"| `{c}` | {np.mean([r['res'][c]['K'] for r in recs]):.0f} | {100 * acc(recs, c)[0]:.1f} | "
                 f"{100 * acc(recs, c, 's')[0]:.1f} |")
    pairs = [("closure_geo", "closure"), ("closure_query", "closure"), ("closure_geo", "values")]
    for r in RHOS:
        pairs += [(f"G_attn@{r}", f"B_attn@{r}"), (f"G_attn_geo@{r}", f"B_attn@{r}"), (f"G_attn@{r}", f"C_attn@{r}"),
                  (f"C_attn_geo@{r}", f"C_attn@{r}"), (f"C_attn_geo@{r}", f"B_attn@{r}"),
                  (f"C_gold_geo@{r}", f"C_gold@{r}"), (f"C_gold_query@{r}", f"C_gold@{r}")]
    for key, nm in (("v", "value acc"), ("s", "scale acc")):
        L += ["", f"### Hiệu cặp — {nm} (điểm %, CI 95% bootstrap theo tài liệu)", "",
              "| Cặp | " + " | ".join(f"{g} (n={len(G[g])})" for g in G) + " |", "|---|" + "---|" * len(G)]
        for a, b in pairs:
            L.append(f"| `{a}` − `{b}` | " + " | ".join(fmt(boot(G[g], a, b, key)) for g in G) + " |")
    (ROOT / "results" / f"e2_{tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen25vl7b")
    main(ap.parse_args().tag)
