"""Draw facts (red) and their heuristic closure (row=green, col=blue, unit=orange) for a few items."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evc.closure import relations  # noqa: E402
from evc.data import DATA, is_num, load_doc, page_words  # noqa: E402

out = Path(__file__).resolve().parents[1] / "results" / "viz"
out.mkdir(parents=True, exist_ok=True)
items = json.load(open(DATA / "e1_set.json"))
idx = [int(a) for a in sys.argv[1:]] or list(range(8))
for k in idx:
    it = items[k]
    words, size = page_words(load_doc(it["uid"]))
    img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
    d = ImageDraw.Draw(img)
    for s in it["fact_ids"]:
        if not is_num(words[s]["text"]):
            continue
        r = relations(words, s, size[0])
        for key, col in (("row", "green"), ("col", "blue"), ("unit", "orange")):
            for i in r[key]:
                d.rectangle(words[i]["bbox"], outline=col, width=3)
    for s in it["fact_ids"]:
        d.rectangle(words[s]["bbox"], outline="red", width=4)
    d.text((10, 10), f"Q: {it['question']}  A: {it['answer']} [{it['scale']}]", fill="red")
    img.save(out / f"{k:03d}.png")
    print(k, it["question"], it["answer"], it["scale"])
