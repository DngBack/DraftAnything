"""Build the E1 question set and render pages at OCR resolution."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evc.data import build_set, render_page, DATA

items, total = build_set(n=600, seed=0)
for it in items:
    render_page(it["uid"], tuple(it["size"]))
out = DATA / "e1_set.json"
json.dump(items, open(out, "w"), indent=1)
from collections import Counter
print("eligible", total, "chosen", len(items), "docs", len({i['uid'] for i in items}))
print(Counter(i["answer_type"] for i in items), Counter(i["scale"] for i in items),
      Counter("multi" if i["n_facts"] > 1 else "single" for i in items))
