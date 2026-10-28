"""Exploratory (2026-10-16, not pre-registered): hidden-state visual guidance vs logit-space visual contrast.

For each item: greedy decoding up to the decision step (as diag_gen.py), then the decision prefix is re-run with the
real image and with a grey image of the same size (same tokens, same positions). Logit-space contrast (VCD-style,
s_b + g (s_r - s_b)) can only move the candidate logits along U_C (h_r - h_b); hidden-state guidance replaces the
layer-l output of the text positions after the image by h_b + g (h_r - h_b) and lets the remaining layers act.
No labels, no probe. Saves candidate scores for every (layer, gamma, scope).

usage: python explore_hcfg.py --model qwen3vl4b --set illusions [--limit N]
"""
import argparse, os, numpy as np, torch
from PIL import Image
from transformers import LogitsProcessorList, StoppingCriteriaList
from pilot import Runner, OUT
from gen_calib import items, Decide
from diag_gen import SETS, Stop
from diag_blank import item_list

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--one_res", type=int, default=1, help="VLMBias: keep one resolution per image")
a = ap.parse_args()
GAMMAS = [0.0, 1.5, 2.0, 3.0, 5.0]
FRACS = [0.25, 0.5, 0.75, 0.9]

R = Runner(a.model, ["x"])
tok = R.p.tokenizer
nL = len(R.layers)
LAYERS = sorted({int(f * nL) for f in FRACS})
task, groups = SETS[a.set]
its = item_list(a.set, items(task))  # diag_gen order; "idx" below indexes this list (= states["idx"])
for j, it in enumerate(its):
    it["idx"] = j
if a.one_res and task == "vlmbias":  # first copy of each (image, wording); the same image comes at 3 resolutions
    seen, keep = set(), []
    for it in its:
        k = (it["key"], it["q"])
        if k not in seen:
            seen.add(k), keep.append(it)
    its = keep
its = its[: a.limit]


def cand_ids(cands, paren=False):
    pre = ("", " ", "(", " (") if paren else ("", " ")
    norm = (lambda t: t.strip().lstrip("(").lower()) if paren else (lambda t: t.strip().lower())
    return [sorted({i for p in pre for v in (c, c.lower(), c.capitalize())
                    for i in tok.encode(p + v, add_special_tokens=False) if norm(tok.decode([i])) == c.lower()}) for c in cands]


digit_ids = sorted({i for d in range(10) for p in ("", " ") for i in tok.encode(p + str(d), add_special_tokens=False)
                    if tok.decode([i]).strip() == str(d)})


def build(it, image):
    msgs = ([{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []) + \
           [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": it["q"]}]}]
    return R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")


@torch.no_grad()
def one(it):
    x = build(it, it["image"])
    cids = cand_ids(it["cands"], it.get("gate") == "letter")
    d = Decide(cids, None, digit_ids if it["cands"][0].isdigit() else None, ())
    n0 = x["input_ids"].shape[1]
    if it.get("gate") == "{":
        d.ready = lambda ids: tok.decode(ids[0, n0:]).strip() == "" or tok.decode(ids[0, n0:][-4:]).rstrip().endswith("{")
    elif it.get("gate") == "letter":
        d.ready = lambda ids: tok.decode(ids[0, n0:]).strip().strip("*").strip() in ("", "(", "Answer:", "Answer: (")
    o = R.m.generate(**x, max_new_tokens=384 if it.get("gate") == "{" else 24, do_sample=False,
                     logits_processor=LogitsProcessorList([d]), stopping_criteria=StoppingCriteriaList([Stop(d)]),
                     return_dict_in_generate=True)
    if not d.done:
        return None
    pre = o.sequences[:, n0:-1]  # tokens generated before the decided one
    grey = Image.new("RGB", it["image"].size, (128, 128, 128))
    xb = build(it, grey)
    assert torch.equal(xb["input_ids"], x["input_ids"])
    def extend(inp):  # per-token inputs (input_ids, attention_mask, mm_token_type_ids, ...) get the text prefix appended
        out = dict(inp)
        for k, v in inp.items():
            if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, n0):
                out[k] = torch.cat([v, pre if k == "input_ids" else (torch.ones_like(pre) if k == "attention_mask"
                                                                    else torch.zeros_like(pre, dtype=v.dtype))], 1)
        return out
    xr, xb = extend(x), extend(xb)
    post = (xr["input_ids"][0] == R.img_id).nonzero().max().item() + 1  # first text position after the image

    def scores(inp, edits=None, hidden=False):
        hooks = []
        for L, fn in (edits or {}).items():
            def hook(mod, args, out, fn=fn):
                return (fn(out[0]),) + out[1:] if isinstance(out, tuple) else fn(out)
            hooks.append(R.layers[L].register_forward_hook(hook))
        try:
            oo = R.m(**inp, output_hidden_states=hidden)
        finally:
            for h in hooks:
                h.remove()
        lg = oo.logits[0, -1].float()
        return np.array([float(lg[i].max()) for i in cids]), oo.hidden_states

    sr, hr = scores(xr, hidden=True)
    sb, hb = scores(xb, hidden=True)
    out = {"raw_gen": d.scores, "real": sr, "blank": sb}
    for L in LAYERS:
        Hr, Hb = hr[L + 1][0], hb[L + 1][0]  # hidden_states[L+1] = output of decoder layer L
        for g in GAMMAS[1:]:
            for scope in ("last", "post"):
                def fn(h, Hr=Hr, Hb=Hb, g=g, scope=scope):
                    h = h.clone()
                    sl = slice(h.shape[1] - 1, None) if scope == "last" else slice(post, None)
                    h[0, sl] = (Hb[sl] + g * (Hr[sl] - Hb[sl])).to(h.dtype)
                    return h
                out[f"h_L{L}_g{g}_{scope}"] = scores(xr, {L: fn})[0]
    return out


rows = []
for j, it in enumerate(its):
    if it["y"] not in it["cands"]:
        continue
    r = one(it)
    if r is None:
        continue
    rows.append((it["cands"].index(it["y"]), str(it.get("key", it.get("pair", j))), it["group"], r, it["idx"]))
    if j % 50 == 0:
        print(j, len(rows), flush=True)
y = np.array([r[0] for r in rows])
keys = sorted(rows[0][3])
os.makedirs(f"{OUT}/explore_hcfg", exist_ok=True)
np.savez(f"{OUT}/explore_hcfg/{a.model}_{a.set}.npz", y=y, key=np.array([r[1] for r in rows]),
         group=np.array([r[2] for r in rows]), idx=np.array([r[4] for r in rows]), names=np.array(keys), S=np.stack([np.stack([r[3][k] for k in keys]) for r in rows]))
acc = lambda s: float(np.mean(s.argmax(1) == y))
S = {k: np.stack([r[3][k] for r in rows]) for k in keys}
print(f"{a.model} {a.set} n={len(y)} raw_gen={acc(S['raw_gen']):.3f} real={acc(S['real']):.3f} blank={acc(S['blank']):.3f}")
for g in GAMMAS[1:]:
    print(f"  logit-contrast g={g}: {acc(S['blank'] + g * (S['real'] - S['blank'])):.3f}")
for k in keys:
    if k.startswith("h_"):
        print(f"  {k}: {acc(S[k]):.3f}")
