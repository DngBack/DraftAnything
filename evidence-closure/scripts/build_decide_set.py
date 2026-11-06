"""Protocol D set: 500 target cells on documents unused by H3, ATTR and the pilots, each with a decoy cell."""
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evc.closure import YEAR_RE, relations  # noqa: E402
from evc.data import DATA, build_set, is_num, load_doc, page_words, render_page  # noqa: E402
from evc.stimuli import digits  # noqa: E402

N = 500
used = {i["uid"] for f in ("h3_train_set.json", "attr_set.json", "percept_set.json") for i in json.load(open(DATA / f))}
items, total = build_set(n=100000, seed=17, split="train")
rng = random.Random(17)
rest = [i for i in items if i["uid"] not in used]
rng.shuffle(rest)
z = zipfile.ZipFile(DATA / "tatdqa_docs_train.zip")
names = set(z.namelist())
seen, out = set(), []
for it in rest:
    if it["uid"] in seen:
        continue
    words, (W, H) = page_words(load_doc(it["uid"], "train"))
    page = [digits(w["text"]) for w in words if is_num(w["text"])]
    for fid in it["fact_ids"]:
        t = words[fid]["text"]
        d = digits(t)
        if not is_num(t) or len(d) < 3 or page.count(d) != 1:
            continue
        r = relations(words, fid, W)
        if not r["row"] or not r["col"]:
            continue
        decoys = [i for i, w in enumerate(words) if i != fid and is_num(w["text"]) and len(digits(w["text"])) >= 3 and page.count(digits(w["text"])) == 1]
        if not decoys:
            continue
        dc = rng.choice(decoys)
        pos = rng.randint(1, len(d) - 2)
        alts = [c for c in "0123456789" if c != d[pos]]
        rng.shuffle(alts)
        out.append(dict(qid=it["qid"], uid=it["uid"], question=it["question"], answer_type=it["answer_type"], size=it["size"], word=t,
                        bbox=words[fid]["bbox"], row=" ".join(words[j]["text"] for j in r["row"]), col=" ".join(words[j]["text"] for j in r["col"]),
                        pos=pos, d_s=alts[0], d_a=alts[1], decoy_word=words[dc]["text"], decoy_bbox=words[dc]["bbox"],
                        num_bboxes=[w["bbox"] for i, w in enumerate(words) if is_num(w["text"]) and (i == fid or not YEAR_RE.match(w["text"]))]))
        seen.add(it["uid"])
        break
    if len(out) == N:
        break
for it in out:
    pdf = DATA / "docs_train" / "train" / f"{it['uid']}.pdf"
    if not pdf.exists():
        nm = next(n for n in (f"train/{it['uid']}.pdf", f"{it['uid']}.pdf") if n in names)
        pdf.write_bytes(z.read(nm))
    render_page(it["uid"], tuple(it["size"]), split="train")
json.dump(out, open(DATA / "decide_set.json", "w"))
print("eligible", total, "unused", len(rest), "chosen", len(out))
