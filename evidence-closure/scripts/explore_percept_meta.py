"""EXPLORATORY, written after seeing pilot B (not in docs/protocol_percept_cf.md): is the log-prob signal more than font size or length?"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_percept import LEVELS, auroc  # noqa: E402
from evc.data import DATA  # noqa: E402
from evc.stimuli import digits, swap_digit  # noqa: E402
from pilot_percept import dig, nums  # noqa: E402

S = {i["qid"]: i for i in json.load(open(DATA / "percept_set.json"))}
L = ["# Thăm dò sau pilot B: log-prob có hơn cỡ chữ và độ dài số không?", ""]
for tag in ("qwen25vl7b", "qwen3vl8b"):
    rows = [json.loads(l) for l in open(ROOT / "results" / f"percept_{tag}.jsonl")]
    rec = []
    for r in rows:
        if r["prompt"] != "copy":
            continue
        it = S[r["qid"]]
        target = dig(swap_digit(it["word"], it["pos"], it["d_s"])) if r["cond"] == "cf" else dig(it["word"])
        rec.append(dict(q=r["qid"], lv=r["level"], cond=r["cond"], ok=target in set(nums(r["text"])), lp=r["lp"], lpmin=r["lpmin"],
                        h=it["bbox"][3] - it["bbox"][1], n=len(digits(it["word"]))))
    L += [f"## {tag}", "", "AUROC theo từng mức, ảnh orig (dự đoán `copy` đọc đúng):", "", "| mức | log-prob TB | log-prob min | chiều cao chữ | số chữ số |", "|---|---|---|---|---|"]
    for lv in LEVELS:
        R = [x for x in rec if x["lv"] == lv and x["cond"] == "orig"]
        y = np.array([x["ok"] for x in R])
        L.append(f"| {lv} | " + " | ".join(f"{auroc(np.array([s * x[k] for x in R]), y):.2f}" for k, s in (("lp", 1), ("lpmin", 1), ("h", 1), ("n", -1))) + " |")
    # within-cell test: same question and level, renderings differ, one read right and one read wrong
    L += ["", "Cặp bất đồng trong cùng ô, cùng mức (orig/null/cf): bản đọc đúng có log-prob cao hơn bản đọc sai không?", "", "| mức | số cặp | tỷ lệ đúng > sai, log-prob TB | log-prob min | chênh TB (nat) |", "|---|---|---|---|---|"]
    all_pairs = []
    for lv in LEVELS + ("all",):
        P = all_pairs if lv == "all" else []
        if lv != "all":
            by = {}
            for x in rec:
                if x["lv"] == lv:
                    by.setdefault(x["q"], []).append(x)
            for q, xs in by.items():
                for a in xs:
                    for b in xs:
                        if a["ok"] and not b["ok"]:
                            P.append((q, a["lp"] - b["lp"], a["lpmin"] - b["lpmin"]))
            all_pairs += P
        if not P:
            L.append(f"| {lv} | 0 | – | – | – |")
            continue
        d, m = np.array([p[1] for p in P]), np.array([p[2] for p in P])
        L.append(f"| {lv} | {len(P)} | {100 * np.mean(d > 0):.0f} % | {100 * np.mean(m > 0):.0f} % | {d.mean():+.3f} |")
    qs = sorted({p[0] for p in all_pairs})
    per = {q: [p[1] > 0 for p in all_pairs if p[0] == q] for q in qs}
    rng = np.random.default_rng(0)
    bs = [np.mean([np.mean(per[qs[k]]) for k in rng.integers(0, len(qs), len(qs))]) for _ in range(2000)]
    L += ["", f"Trung bình theo câu, tỷ lệ đúng > sai: {100 * np.mean([np.mean(v) for v in per.values()]):.0f} % "
          f"[{100 * np.percentile(bs, 2.5):.0f}, {100 * np.percentile(bs, 97.5):.0f}], {len(qs)} câu có cặp bất đồng.", ""]
    # what do UNREADABLE-free wrong outputs look like: distance of wrong read from the target
    near = []
    for x, r in zip(rec, [r for r in rows if r["prompt"] == "copy"]):
        if x["lv"] in ("s40", "s30") and x["cond"] == "orig" and not x["ok"]:
            near.append(r["text"][:24] + " ← " + S[r["qid"]]["word"])
    L += ["Ví dụ đọc sai, ảnh orig, s40/s30: " + " · ".join(near[:8]), ""]
out = ROOT / "results" / "percept_meta_explore.md"
out.write_text("\n".join(L))
print("\n".join(L))
