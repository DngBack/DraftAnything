"""E3 confirmation set: TAT-DQA test (gold), same filter and sampling as E1. Not inspected."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evc.data import build_set, render_page, DATA

items, total = build_set(n=600, seed=0, split="test", qfile="tatdqa_dataset_test_gold.json")
for it in items:
    render_page(it["uid"], tuple(it["size"]), split="test")
json.dump(items, open(DATA / "e3_test_set.json", "w"), indent=1)
print("eligible", total, "chosen", len(items), "docs", len({i['uid'] for i in items}))
print(Counter(i["answer_type"] for i in items), Counter("multi" if i["n_facts"] > 1 else "single" for i in items))
