"""H3b test2: every eligible TAT-DQA test question NOT in e3_test_set.json (same filter as E1/E3). Not inspected."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evc.data import build_set, render_page, DATA

items, total = build_set(n=100000, seed=2, split="test", qfile="tatdqa_dataset_test_gold.json")
used = {i["qid"] for i in json.load(open(DATA / "e3_test_set.json"))}
items = [i for i in items if i["qid"] not in used]
for it in items:
    render_page(it["uid"], tuple(it["size"]), split="test")
json.dump(items, open(DATA / "h3b_test2_set.json", "w"), indent=1)
print("eligible", total, "test2", len(items), "docs", len({i['uid'] for i in items}), Counter(i["answer_type"] for i in items))
