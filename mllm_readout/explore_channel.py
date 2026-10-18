"""Exploratory (2026-10-18): channel switching, a zero-label repair for errors where the evidence is seen but the verbal
judgement is overridden. Stage 1 (with image): the model is asked not to answer but to report the geometry relevant
to the question as coordinates. Stage 2 (same model, text only, no image): it answers from those numbers.
Illusions, one view per (image, wording). Output: results/g4/<model>_illusions_channel.npz (idx, y, pred, text).
"""
import argparse, os, re, numpy as np, torch
from pilot import Runner, OUT
from gen_calib import items
from diag_blank import item_list

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--limit", type=int, default=10 ** 9)
a = ap.parse_args()
R = Runner(a.model, ["x"])
its = item_list("illusions", items("vlmbias"))
test, seen = [], set()
for j, it in enumerate(its):
    k = (it["key"], it["q"])
    if k not in seen:
        seen.add(k), test.append(j)
test = test[:: max(1, len(test) // a.limit)][: a.limit] if a.limit < len(test) else test

S1 = ('Locate every object mentioned in this question: "{q}". Do not answer the question. Output the bounding boxes '
      'in JSON format: [{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}].')
S2 = ("Bounding boxes [x1, y1, x2, y2] of the objects in an image:\n{m}\n\nUsing only these measurements, answer the "
      "question: {q}\nFirst compute the relevant lengths, sizes or positions, then give the final answer in curly "
      "brackets, e.g., {{Yes}} or {{No}}.")


def gen(content, n):
    x = R.p.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
                                return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = R.m.generate(**x, max_new_tokens=n, do_sample=False)
    return R.p.decode(o[0, x["input_ids"].shape[1]:], skip_special_tokens=True)


pred, ys, texts = [], [], []
for n, j in enumerate(test):
    it = its[j]
    core = it["q"].split(" Answer in curly")[0]
    m = gen([{"type": "image", "image": it["image"]}, {"type": "text", "text": S1.format(q=core)}], 256)
    t = gen([{"type": "text", "text": S2.format(m=m.strip(), q=core)}], 768)
    ans = re.findall(r"\{\s*(Yes|No)\s*\}", t, re.I)
    p = ans[-1].capitalize() if ans else ""
    pred.append(it["cands"].index(p) if p in it["cands"] else -1)
    ys.append(it["cands"].index(it["y"]))
    texts.append(m + "\n=====\n" + t)
    if n % 25 == 0:
        print(n, flush=True)
pred, ys = np.array(pred), np.array(ys)
np.savez(f"{OUT}/g4/{a.model}_illusions_channel.npz", idx=np.array(test), y=ys, pred=pred, text=np.array(texts))
print(f"{a.model} channel: acc {np.mean(pred == ys):.3f} unparsed {np.mean(pred < 0):.3f}", flush=True)
