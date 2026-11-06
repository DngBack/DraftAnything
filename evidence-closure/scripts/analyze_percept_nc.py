"""Pilot C analysis (docs/protocol_percept_nc.md): beam coverage of the true number when the top hypothesis is wrong."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.data import DATA  # noqa: E402
from pilot_percept import dig, ed, nums  # noqa: E402

S = {i["qid"]: i for i in json.load(open(DATA / "percept_set.json"))}
L = ["# Pilot C: độ phủ beam", "", "| model | mức | top-1 đúng | top-8 có số đúng | độ phủ khi top-1 sai [CI] | n top-1 sai | sai top-1 cách số đúng ≤ 1 chữ số |", "|---|---|---|---|---|---|---|"]
for tag in ("qwen25vl7b", "qwen3vl8b"):
    R = [json.loads(l) for l in open(ROOT / "results" / f"percept_nc_{tag}.jsonl")]
    for lv in ("s50", "s40", "s30"):
        top1, anyk, near = [], [], []
        for r in (x for x in R if x["level"] == lv):
            t = dig(S[r["qid"]]["word"])
            o = int(np.argmax(r["scores"]))
            g1 = nums(r["texts"][o])
            top1.append(t in g1)
            anyk.append(any(t in nums(x) for x in r["texts"]))
            near.append(bool(g1) and min(ed(t, g) for g in g1) <= 1)
        top1, anyk, near = map(np.array, (top1, anyk, near))
        w = ~top1
        rng = np.random.default_rng(0)
        cov = anyk[w]
        b = [cov[rng.integers(0, len(cov), len(cov))].mean() for _ in range(2000)] if len(cov) else [np.nan]
        L.append(f"| {tag} | {lv} | {100 * top1.mean():.1f} | {100 * anyk.mean():.1f} | {100 * cov.mean():.1f} [{100 * np.percentile(b, 2.5):.1f}, {100 * np.percentile(b, 97.5):.1f}] | {w.sum()} | {100 * near[w].mean():.1f} |")
out = ROOT / "results" / "percept_nc.md"
out.write_text("\n".join(L))
print("\n".join(L))
