"""G5 N (docs/g5_protocol.md): native visual-pathway features, question-blind, in diag_blank.item_list order.

For each item: the input of the model's own projector (vision-encoder output, "pre"), mean over patches, and the
projector output entering LLM layer 0 at image positions ("l0"), mean over image tokens. The LLM forward stops at
layer 0. Output: results/explore_native/<model>_<set>.npz with pre (n, d_vis), l0 (n, d_llm).

usage: python explore_native.py --model qwen3vl4b --sets illusions countbench
"""
import argparse, os, numpy as np, torch
from gen_calib import items
from diag_gen import SETS
from diag_blank import item_list
from pilot import Runner, OUT

D = f"{OUT}/explore_native"


class Stop(Exception):
    pass


def projector(m):
    for n, mod in m.named_modules():
        if n.endswith(("multi_modal_projector", "visual.merger")) or n == "mlp1" or n.endswith(".mlp1"):
            return n, mod
    raise ValueError("no projector module found")


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--sets", nargs="+", required=True)
a = ap.parse_args()
os.makedirs(D, exist_ok=True)
R = Runner(a.model, ["x"])
name, proj = projector(R.m)
print("projector:", name, flush=True)
cap = {}


def pre_hook(mod, args, kwargs):
    x = args[0] if args else next(iter(kwargs.values()))
    cap.setdefault("pre", []).append(x.detach().float().reshape(-1, x.shape[-1]))


def l0_hook(mod, args, kwargs):
    cap["l0"] = (args[0] if args else kwargs["hidden_states"])[0].detach().float()
    raise Stop


proj.register_forward_pre_hook(pre_hook, with_kwargs=True)
R.layers[0].register_forward_pre_hook(l0_hook, with_kwargs=True)


@torch.no_grad()
def feats(it):
    cap.clear()
    msgs = [{"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]}]
    x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    try:
        R.m(**x)
    except Stop:
        pass
    im = x["input_ids"][0] == R.img_id
    return torch.cat(cap["pre"]).mean(0).cpu().half(), cap["l0"][im].mean(0).cpu().half()


cache = {}
for s in a.sets:
    out = f"{D}/{a.model}_{s}.npz"
    if os.path.exists(out):
        continue
    task = SETS[s][0]
    if task not in cache:
        cache = {task: items(task)}
    its = item_list(s, cache[task])
    memo, P, L = {}, [], []
    for it in its:  # question-blind: one pass per image
        k = id(it["image"])
        if k not in memo:
            memo[k] = feats(it)
        P.append(memo[k][0]), L.append(memo[k][1])
    np.savez(out, pre=torch.stack(P).numpy(), l0=torch.stack(L).numpy())
    print(a.model, s, tuple(P[0].shape), tuple(L[0].shape), len(P), flush=True)
