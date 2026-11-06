"""Render which visual tokens each condition keeps (kept cells shown, dropped cells greyed).

usage: viz_tokens.py <item_index> <cond> [<cond> ...]   (needs a GPU for the attention scores)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.closure import Grid  # noqa: E402
from evc.data import DATA, load_doc, page_words  # noqa: E402
from evc.vlm import VLM  # noqa: E402
from run_e1 import conditions  # noqa: E402

k = int(sys.argv[1])
names = sys.argv[2:]
it = json.load(open(DATA / "e1_set.json"))[k]
vlm = VLM(str(ROOT / "hf" / "Qwen2.5-VL-3B-Instruct"))
words, size = page_words(load_doc(it["uid"]))
img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
inp = vlm.prepare(img, it["question"])
emb, pos = vlm.encode(inp)
sc = vlm.scores(inp, emb, pos)
grid = Grid(size, inp.grid)
conds, _, _, _ = conditions(words, grid, sc, it["fact_ids"], size[0])
conds["attention"] = None
a = np.asarray(img).astype(np.float32)
panels = []
for n in names:
    if n == "attention":
        m = (sc / sc.max()).reshape(grid.gh, grid.gw)
    else:
        m = np.zeros(grid.n)
        m[conds[n]] = 1
        m = m.reshape(grid.gh, grid.gw)
    M = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(size, Image.NEAREST)) / 255.0
    panels.append((a * (0.15 + 0.85 * M[..., None]) + 255 * 0.85 * (1 - M[..., None]) * 0.6).clip(0, 255).astype(np.uint8))
out = ROOT / "results" / "viz" / f"tokens_{k:03d}.png"
Image.fromarray(np.concatenate(panels, 1)).save(out)
print(out, it["question"], it["answer"])
