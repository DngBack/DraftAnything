"""H3c confirmation set: every eligible TAT-DQA dev question NOT in e1_set.json (same filter). Not inspected."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evc.data import build_set, render_page, DATA

items, total = build_set(n=100000, seed=3, split="dev")
used = {i["qid"] for i in json.load(open(DATA / "e1_set.json"))}
items = [i for i in items if i["qid"] not in used]
for it in items:
    render_page(it["uid"], tuple(it["size"]), split="dev")
json.dump(items, open(DATA / "h3c_dev2_set.json", "w"), indent=1)
print("eligible", total, "dev2", len(items), "docs", len({i['uid'] for i in items}), Counter(i["answer_type"] for i in items))
