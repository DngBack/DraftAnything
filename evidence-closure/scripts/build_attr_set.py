"""Attribution set: eligible TAT-DQA train questions on documents NOT in h3_train_set.json (never used by any policy)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evc.data import DATA, build_set, render_page  # noqa: E402

used = {i["uid"] for i in json.load(open(DATA / "h3_train_set.json"))}
items, total = build_set(n=100000, seed=11, split="train")
rest = [i for i in items if i["uid"] not in used]
import random  # noqa: E402

random.Random(11).shuffle(rest)
# one question per document first, so that documents are spread; 400 questions
seen, a, b = set(), [], []
for i in rest:
    (b if i["uid"] in seen else a).append(i)
    seen.add(i["uid"])
chosen = (a + b)[:400]
import zipfile  # noqa: E402

z = zipfile.ZipFile(DATA / "tatdqa_docs_train.zip")
names = set(z.namelist())
for it in chosen:
    pdf = DATA / "docs_train" / "train" / f"{it['uid']}.pdf"
    if not pdf.exists():
        nm = next(n for n in (f"train/{it['uid']}.pdf", f"{it['uid']}.pdf") if n in names)
        pdf.write_bytes(z.read(nm))
    render_page(it["uid"], tuple(it["size"]), split="train")
json.dump(chosen, open(DATA / "attr_set.json", "w"), indent=1)
print("eligible", total, "doc-disjoint", len(rest), "chosen", len(chosen), "docs", len({i["uid"] for i in chosen}),
      Counter(i["answer_type"] for i in chosen))
