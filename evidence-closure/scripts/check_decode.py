"""Sanity: custom batched decode with ALL tokens kept must match HF model.generate (greedy).

usage: check_decode.py <model_path> [n_items]
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evc.data import DATA  # noqa: E402
from evc.vlm import load_vlm  # noqa: E402

vlm = load_vlm(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
items = json.load(open(DATA / "e1_set.json"))[:n]
same = 0
for it in items:
    img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
    inp = vlm.prepare(img, it["question"])
    emb, pos = vlm.encode(inp)
    ours = vlm.answer_batch(emb, pos, inp, [None])[0]
    with torch.no_grad():
        g = vlm.model.generate(**{k: v for k, v in inp.items()}, max_new_tokens=48, do_sample=False, repetition_penalty=1.0, top_k=None, top_p=None, temperature=None)
    ref = vlm.tok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    same += ours == ref
    print(f"{ours == ref!s:5}  ours={ours[:50]!r}  hf={ref[:50]!r}")
print(f"match {same}/{len(items)}")
