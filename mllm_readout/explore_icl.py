"""Exploratory (2026-10-18): can the MLLM use the labels a linear readout uses? In-context k-shot on illusions.

Test items: one view per (image, wording) (264). For each, k demonstrations per class are drawn (seeded by item) from
items with the same wording and a different base image; each demonstration is a user turn (image + question) and an
assistant turn with the gold answer in braces. Greedy, 16 new tokens, G3 parser.
Output: results/g4/icl_<model>_k<k>.npz (idx, y, pred).
G5 C (docs/g5_protocol.md): k = 0 (free generation, same parser) and --fmt variants -> icl_<model>_<fmt>_k<k>.npz:
instr (an instruction that the examples share the question and the answer depends on an exact measurement), alt
(alternating classes, random start), single (all examples in one user turn as numbered images with their answers).
"""
import argparse, os, re, numpy as np, torch
from pilot import Runner, OUT
from gen_calib import items, parse
from diag_blank import item_list

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--fmt", default="plain", choices=["plain", "instr", "alt", "single"])
a = ap.parse_args()
R = Runner(a.model, ["x"])
its = item_list("illusions", items("vlmbias"))
base = [re.sub(r"_px\d+", "", str(it["key"])) for it in its]
test, seen = [], set()
for j, it in enumerate(its):
    k = (it["key"], it["q"])
    if k not in seen:
        seen.add(k), test.append(j)


INSTR = ("The examples below ask the same question about different images. The correct answer depends on an exact "
         "measurement of the lines in the image, not on how they appear. ")


def turn(it, pre=""):
    return {"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": pre + it["q"]}]}


for k in a.ks:
    out = f"{OUT}/g4/icl_{a.model}{'' if a.fmt == 'plain' else '_' + a.fmt}_k{k}.npz"
    if os.path.exists(out):
        continue
    pred, ys = [], []
    for n, j in enumerate(test):
        it, rng = its[j], np.random.default_rng(j)
        msgs = [{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []
        demos = []
        for c in it["cands"]:
            pool = [i for i, o in enumerate(its) if o["q"] == it["q"] and o["y"] == c and base[i] != base[j]]
            demos.append(list(rng.choice(pool, k, replace=False)) if k else [])
        if a.fmt == "alt":
            first = int(rng.integers(2))
            order = [demos[(first + t) % 2][t // 2] for t in range(2 * k)]
        else:
            order = list(rng.permutation(demos[0] + demos[1])) if k else []
        pre = INSTR if a.fmt == "instr" else ""
        if a.fmt == "single" and k:
            content = []
            for t, i in enumerate(order):
                content += [{"type": "text", "text": f"Example {t + 1}:"}, {"type": "image", "image": its[i]["image"]},
                            {"type": "text", "text": f"{its[i]['q']} {{{its[i]['y']}}}"}]
            content += [{"type": "text", "text": "Now the new image:"}, {"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]
            msgs.append({"role": "user", "content": content})
        else:
            for t, i in enumerate(order):
                msgs += [turn(its[i], pre if t == 0 else ""), {"role": "assistant", "content": [{"type": "text", "text": "{" + its[i]["y"] + "}"}]}]
            msgs.append(turn(it, pre if not order else ""))
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
    print(f"{a.model} ICL {a.fmt} k={k} per class: acc {np.mean(pred == ys):.3f} unparsed {np.mean(pred < 0):.3f}", flush=True)
