"""Exploratory (2026-10-16): spatial input features for the input-side control, aligned with the G3 item lists.

MLLM: image-token states entering decoder layer 0 (projector output) and layer 4 (after deepstack injection in
Qwen3-VL), average-pooled to a 4x4 grid plus the mean (17 vectors). Image tokens precede the question, so these states
cannot depend on it. External: SigLIP2-base and DINOv2-base patch features, same pooling.
Output: results/explore_input/<source>_<set>.npz with feats (n_items, n_layers, 17, d) in diag_blank.item_list order.

usage: python explore_input.py --src qwen3vl4b --sets illusions countbench mmvp
       python explore_input.py --src siglip2 dinov2 --sets illusions countbench mmvp
"""
import argparse, os, numpy as np, torch
from gen_calib import items
from diag_gen import SETS
from diag_blank import item_list
from pilot import OUT

D = f"{OUT}/explore_input"
EXT = {"siglip2": "google/siglip2-base-patch16-512", "dinov2": "facebook/dinov2-base"}


class Stop(Exception):
    pass


def pool(tok, h, w):  # (h*w, d) row-major -> (17, d): 4x4 average grid + mean
    g = tok.T.reshape(1, -1, h, w).float()
    return torch.cat([torch.nn.functional.adaptive_avg_pool2d(g, 4).flatten(2)[0].T, tok.float().mean(0, keepdim=True)])


def ext_encoder(name):
    from transformers import AutoModel, AutoImageProcessor
    m = AutoModel.from_pretrained(EXT[name], dtype=torch.float32).cuda().eval()
    m = m.vision_model if name == "siglip2" else m
    proc = AutoImageProcessor.from_pretrained(EXT[name])

    @torch.no_grad()
    def f(it):
        x = proc(images=it["image"], return_tensors="pt")["pixel_values"].cuda()
        h = m(pixel_values=x).last_hidden_state[0]
        h = h[1:] if name == "dinov2" else h  # drop CLS
        s = int(round(h.shape[0] ** 0.5))
        return pool(h[: s * s], s, s)[None]
    return f


def mllm_encoder(name, deep=False):
    from pilot import Runner
    R = Runner(name, ["x"])
    cap = {}
    LS = (round(0.75 * len(R.layers)) + 1,) if deep else (0, 4)  # deep: input of layer L+1 = output of the steering layer L (= states "last"[:, L + 1])

    def pre(L):
        def hook(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            cap[L] = h[0].detach()
            if L == max(LS):
                raise Stop
        return hook
    for L in LS:
        R.layers[L].register_forward_pre_hook(pre(L), with_kwargs=True)
    merge = getattr(getattr(R.m.config, "vision_config", None), "spatial_merge_size", None)

    @torch.no_grad()
    def f(it):
        c = [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]
        msgs = [{"role": "user", "content": c[::-1] if QFIRST else c}]  # QFIRST: question before the image -> question-aware image tokens
        x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        try:
            R.m(**x)
        except Stop:
            pass
        im = x["input_ids"][0] == R.img_id
        n = int(im.sum())
        if "image_grid_thw" in x:
            t, h, w = x["image_grid_thw"][0].tolist()
            h, w = h // merge, w // merge
        else:
            h = w = int(round(n ** 0.5))
        assert h * w == n, (h, w, n)
        return torch.stack([pool(cap[L][im], h, w) for L in LS])
    return f


ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--sets", nargs="+", required=True)
ap.add_argument("--deep", action="store_true", help="MLLM image tokens entering layer round(0.75 nL) (question-blind); -> explore_input_deep/")
ap.add_argument("--qfirst", action="store_true")
a = ap.parse_args()
QFIRST = a.qfirst
D = D + ("_deep" if a.deep else "") + ("_qfirst" if a.qfirst else "")
os.makedirs(D, exist_ok=True)
cache = {}
for src in a.src:
    enc = ext_encoder(src) if src in EXT else mllm_encoder(src, a.deep)
    for name in a.sets:
        out = f"{D}/{src}_{name}.npz"
        if os.path.exists(out):
            continue
        task = SETS[name][0]
        if task not in cache:
            cache = {task: items(task)}
        its = item_list(name, cache[task])
        memo, F = {}, []
        for it in its:  # the same image may come with several questions (features are question-free for externals)
            k = (id(it["image"]), it["q"] if src not in EXT else "")  # with QFIRST the image tokens depend on the question
            if k not in memo:
                memo[k] = enc(it).cpu().half()
            F.append(memo[k])
        np.savez(out, feats=torch.stack(F).numpy())
        print(src, name, tuple(F[0].shape), len(F), flush=True)
    del enc
    torch.cuda.empty_cache()
