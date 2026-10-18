"""Exploratory (2026-10-18): can the MLLM use the labels a linear readout uses? In-context k-shot on illusions.

Test items: one view per (image, wording) (264). For each, k demonstrations per class are drawn (seeded by item) from
items with the same wording and a different base image; each demonstration is a user turn (image + question) and an
assistant turn with the gold answer in braces. Greedy, 16 new tokens, G3 parser.
Output: results/g4/icl_<model>_k<k>.npz (idx, y, pred).
"""
import argparse, os, re, numpy as np, torch
from pilot import Runner, OUT
from gen_calib import items, parse
from diag_blank import item_list

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
a = ap.parse_args()
R = Runner(a.model, ["x"])
its = item_list("illusions", items("vlmbias"))
base = [re.sub(r"_px\d+", "", str(it["key"])) for it in its]
test, seen = [], set()
for j, it in enumerate(its):
    k = (it["key"], it["q"])
    if k not in seen:
        seen.add(k), test.append(j)


def turn(it):
    return {"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]}


for k in a.ks:
    out = f"{OUT}/g4/icl_{a.model}_k{k}.npz"
    if os.path.exists(out):
        continue
    pred, ys = [], []
    for n, j in enumerate(test):
        it, rng = its[j], np.random.default_rng(j)
        msgs = [{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []
        demos = []
        for c in it["cands"]:
            pool = [i for i, o in enumerate(its) if o["q"] == it["q"] and o["y"] == c and base[i] != base[j]]
            demos += list(rng.choice(pool, k, replace=False))
        for i in rng.permutation(demos):
            msgs += [turn(its[i]), {"role": "assistant", "content": [{"type": "text", "text": "{" + its[i]["y"] + "}"}]}]
        msgs.append(turn(it))
        x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = R.m.generate(**x, max_new_tokens=16, do_sample=False)
        p = parse(R.p.decode(o[0, x["input_ids"].shape[1]:], skip_special_tokens=True), it["cands"])
        pred.append(it["cands"].index(p) if p in it["cands"] else -1)
        ys.append(it["cands"].index(it["y"]))
        if n % 50 == 0:
            print(k, n, flush=True)
    pred, ys = np.array(pred), np.array(ys)
    np.savez(out, idx=np.array(test), y=ys, pred=pred)
    print(f"{a.model} ICL k={k} per class: acc {np.mean(pred == ys):.3f} unparsed {np.mean(pred < 0):.3f}", flush=True)
