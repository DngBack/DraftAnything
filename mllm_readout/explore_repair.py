"""G4 repairs R4 (compute: thinking mode) and R5 (see more: 2x upscaled image), docs/g4_protocol.md.

Same items and prompts as G3; illusions use one view per (image, wording) (first copy in item order). Greedy, batched
with left padding; the answer is parsed with gen_calib.parse from the text after the reasoning block (R4) or from the
whole output (R5). Output: results/g4/<model>_<set>_<mode>.npz with idx (diag_blank.item_list order), y, pred, text.

usage: python explore_repair.py --model qwen35_4b --mode think --sets illusions countbench mmvp
"""
import argparse, os, numpy as np, torch
from PIL import Image
from pilot import Runner, OUT
from gen_calib import items, parse
from diag_gen import SETS
from diag_blank import item_list

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mode", choices=["think", "res2", "qfirst", "plain"], required=True)  # qfirst: question before the image (question-aware image tokens); plain: control (free generation, G3 prompt)
ap.add_argument("--sets", nargs="+", required=True)
ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--pilot", type=int, default=0, help="only G3 errors + this many G3-correct items (seed 0); file suffix _pilot")
a = ap.parse_args()
D = f"{OUT}/g4"
os.makedirs(D, exist_ok=True)
R = Runner(a.model, ["x"])
R.p.tokenizer.padding_side = "left"


def upscale(im):
    w, h = im.size
    s = min(2.0, 1536 / max(w, h))
    return im.resize((round(w * s), round(h * s)), Image.BICUBIC)


@torch.no_grad()
def batch(its):
    convs = [([{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []) +
             [{"role": "user", "content": [{"type": "image", "image": upscale(it["image"]) if a.mode == "res2" else it["image"]},
                                           {"type": "text", "text": it["q"]}][::-1 if a.mode == "qfirst" else 1]}] for it in its]
    kw = {"enable_thinking": True} if a.mode == "think" else {}
    x = R.p.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                padding=True, **kw).to("cuda")
    long = any(it.get("gate") == "{" for it in its)
    n = 1536 if a.mode == "think" else (384 if long else 24)
    o = R.m.generate(**x, max_new_tokens=n, do_sample=False)
    txt = R.p.batch_decode(o[:, x["input_ids"].shape[1]:], skip_special_tokens=False)
    txt = [t.replace(R.p.tokenizer.pad_token or "\0", "").replace(R.p.tokenizer.eos_token or "\0", "") for t in txt]
    if a.mode != "think":
        return txt, txt
    for i, t in enumerate(txt):  # budget forcing (addendum 1): truncated reasoning is closed and the answer generated
        if "</think>" in t:
            continue
        xi = R.p.apply_chat_template(convs[i: i + 1], add_generation_prompt=True, tokenize=True, return_dict=True,
                                     return_tensors="pt", **kw).to("cuda")
        n0 = xi["input_ids"].shape[1]
        gen = R.p.tokenizer(t.rstrip() + "\n</think>\n\n", add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        for k, v in list(xi.items()):
            if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, n0):
                xi[k] = torch.cat([v, gen if k == "input_ids" else (torch.ones_like(gen) if k == "attention_mask"
                                                                     else torch.zeros_like(gen, dtype=v.dtype))], 1)
        oi = R.m.generate(**xi, max_new_tokens=24, do_sample=False)
        txt[i] = t.rstrip() + "\n</think>\n\n" + R.p.decode(oi[0, xi["input_ids"].shape[1]:], skip_special_tokens=True)
    return [t.split("</think>")[-1] for t in txt], txt


for name in a.sets:
    out = f"{D}/{a.model}_{name}_{a.mode}{'_pilot' if a.pilot else ''}.npz"
    if os.path.exists(out):
        continue
    task = SETS[name][0]
    its = item_list(name, items(task))
    sel, seen = [], set()
    for j, it in enumerate(its):
        if it["y"] not in it["cands"]:
            continue
        k = (str(it.get("key", j)), it["q"])
        if task == "vlmbias" and k in seen:
            continue
        seen.add(k), sel.append(j)
    if a.pilot:  # G3 decisions: all errors + a fixed sample of correct items (fix rates need errors; broken rate the sample)
        g3 = "g3_heldout" if a.model.startswith(("qwen35", "qwen2vl_2b")) else "g3"
        st = np.load(f"{OUT}/{g3}/{a.model}_{name}_states.npz")
        wrong = dict(zip(st["idx"].tolist(), (st["ans"] != st["y"]).tolist()))
        ok = [j for j in sel if j in wrong and not wrong[j]]
        keep = set(j for j in sel if wrong.get(j)) | set(np.random.default_rng(0).permutation(ok)[: a.pilot].tolist())
        sel = [j for j in sel if j in keep]
    sel = sel[: a.limit]
    preds, texts = [], []
    for b in range(0, len(sel), a.bs):
        chunk = [its[j] for j in sel[b: b + a.bs]]
        ans, raw = batch(chunk)
        preds += [parse(t, it["cands"]) for t, it in zip(ans, chunk)]
        texts += raw
        print(name, b + len(chunk), "/", len(sel), flush=True)
    y = np.array([its[j]["cands"].index(its[j]["y"]) for j in sel])
    pred = np.array([its[j]["cands"].index(p) if p in its[j]["cands"] else -1 for j, p in zip(sel, preds)])
    np.savez(out, idx=np.array(sel), y=y, pred=pred, text=np.array(texts))
    print(f"{a.model} {name} {a.mode}: n={len(y)} acc={np.mean(pred == y):.3f} unparsed={np.mean(pred < 0):.3f}", flush=True)
