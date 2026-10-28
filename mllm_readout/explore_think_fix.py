"""Exploratory (2026-10-18): answer extraction for R4 thinking outputs the G3 parser cannot read.

For each item of results/g4/<model>_<set>_think.npz whose answer is not a candidate, the full generated text (reasoning
and free-text answer) is appended to the prompt, followed by "\nFinal answer: {", and up to 8 greedy tokens are
generated; the answer is parsed from "{" + continuation. Parsed items keep their answer.
Output: results/g4/<model>_<set>_thinkfix.npz (same fields as the think file).
"""
import argparse, numpy as np, torch
from pilot import Runner, OUT
from gen_calib import items, parse
from diag_gen import SETS
from diag_blank import item_list

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--sets", nargs="+", required=True)
a = ap.parse_args()
R = Runner(a.model, ["x"])
tok = R.p.tokenizer

for name in a.sets:
    d = dict(np.load(f"{OUT}/g4/{a.model}_{name}_think.npz"))
    its = item_list(name, items(SETS[name][0]))
    pred, text = d["pred"].copy(), d["text"].astype(object).copy()
    todo = np.flatnonzero(pred < 0)
    for k in todo:
        it = its[int(d["idx"][k])]
        msgs = ([{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []) + \
               [{"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]}]
        x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                    enable_thinking=True).to("cuda")
        n0 = x["input_ids"].shape[1]
        gen = tok(str(text[k]).rstrip() + "\nFinal answer: {", add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        for key, v in list(x.items()):
            if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, n0):
                x[key] = torch.cat([v, gen if key == "input_ids" else (torch.ones_like(gen) if key == "attention_mask"
                                                                       else torch.zeros_like(gen, dtype=v.dtype))], 1)
        with torch.no_grad():
            o = R.m.generate(**x, max_new_tokens=8, do_sample=False)
        cont = tok.decode(o[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
        p = parse("{" + cont, it["cands"])
        pred[k] = it["cands"].index(p) if p in it["cands"] else -1
        text[k] = str(text[k]) + "\nFinal answer: {" + cont
    np.savez(f"{OUT}/g4/{a.model}_{name}_thinkfix.npz", idx=d["idx"], y=d["y"], pred=pred, text=text.astype(str))
    print(f"{a.model} {name}: fixed parse {np.sum(pred[todo] >= 0)}/{len(todo)}; acc think {np.mean(d['pred'] == d['y']):.3f} "
          f"-> thinkfix {np.mean(pred == d['y']):.3f}", flush=True)
