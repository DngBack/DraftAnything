"""G7 I (docs/g7_protocol.md, exploratory): intervention along the geometric direction on illusynC.

Image-position states at the input of the layer at 25% depth (prompt pass, mean over image positions) for every
decided item. d+ = mean(r >= .25) - mean(equal), d- = mean(r <= -.25) - mean(equal), estimated on all items except the
240 test items (equal; 40 per family x regime, seed 0). Each test item is generated to the G3 decision step with
alpha * d added at every image position of that layer's input (prompt pass). Conditions: none, d+ / d- at alpha in
{0.5, 1, 2}, a random direction with the norm of d+ at alpha 2. Read: share of "No" (unequal).
Output: results/g7/steer_<model>.npz.   usage: python g7_steer.py --model qwen3vl8b
"""
import argparse, os, numpy as np, torch
from transformers import LogitsProcessorList, StoppingCriteriaList
from gen_calib import items, Decide
from diag_gen import Stop
from diag_blank import item_list
from g5_swap import Halt
from pilot import OUT, Runner

D = f"{OUT}/g7"
ALPHAS = (0.5, 1.0, 2.0)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
LIMIT = int(os.environ.get("STEER_LIMIT", 10 ** 9))  # smoke tests only (output suffix _smoke)
out = f"{D}/steer_{a.model}{'_smoke' if LIMIT < 10 ** 9 else ''}.npz"
if os.path.exists(out):
    raise SystemExit
R = Runner(a.model, ["x"])
tok = R.p.tokenizer
nL = len(R.layers)
l = int(round(0.25 * nL))
its = item_list("illusynC", items("illusynC"))
st = np.load(f"{OUT}/g6_syn/{a.model}_illusynC_states.npz", allow_pickle=True)
idx = st["idx"]
r = np.array([its[i]["logr"] for i in idx])
fam = np.array([its[i]["group"] for i in idx])
reg = np.array([its[i]["regime"] for i in idx])
rng = np.random.default_rng(0)
test = np.concatenate([rng.choice(np.flatnonzero((r == 0) & (fam == f) & (reg == g)), 40, replace=False)
                       for f in sorted(set(fam)) for g in "AB"])


def enc(it):
    msgs = [{"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]}]
    return R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")


@torch.no_grad()
def capture(it):
    x, cap = enc(it), {}
    im = x["input_ids"][0] == R.img_id

    def h(mod, args, kwargs):
        cap["h"] = (args[0] if args else kwargs["hidden_states"])[0, im].float().mean(0).cpu()
        raise Halt
    hk = R.layers[l].register_forward_pre_hook(h, with_kwargs=True)
    try:
        R.m(**x)
    except Halt:
        pass
    finally:
        hk.remove()
    return cap["h"]


cids = [sorted({t for p in ("", " ") for v in (c, c.lower()) for t in tok.encode(p + v, add_special_tokens=False)
                if tok.decode([t]).strip().lower() == c.lower()}) for c in ("Yes", "No")]


@torch.no_grad()
def generate(it, vec=None):
    x = enc(it)
    im = x["input_ids"][0] == R.img_id
    n0 = x["input_ids"].shape[1]
    dd = Decide(cids, None, None, ())
    dd.ready = lambda ids, n0=n0: tok.decode(ids[0, n0:]).strip() == "" or tok.decode(ids[0, n0:][-4:]).rstrip().endswith("{")
    hk = None
    if vec is not None:
        def h(mod, args, kwargs):
            hh = args[0] if args else kwargs["hidden_states"]
            if hh.shape[1] >= n0:
                hh[0, :n0][im] += vec.to(hh.device, hh.dtype)
        hk = R.layers[l].register_forward_pre_hook(h, with_kwargs=True)
    try:
        R.m.generate(**x, max_new_tokens=384, do_sample=False, logits_processor=LogitsProcessorList([dd]),
                     stopping_criteria=StoppingCriteriaList([Stop(dd)]))
    finally:
        if hk is not None:
            hk.remove()
    return (int(np.argmax(dd.scores)), np.asarray(dd.scores, np.float32)) if dd.done else (-1, np.full(2, np.nan, np.float32))


H = torch.stack([capture(its[i]) for i in idx])
est = np.ones(len(idx), bool)
est[test] = False
mu0 = H[est & (r == 0)].mean(0)
dp, dm = H[est & (r >= 0.25)].mean(0) - mu0, H[est & (r <= -0.25)].mean(0) - mu0
g = torch.randn(dp.shape, generator=torch.Generator().manual_seed(0))
dr = g / g.norm() * dp.norm()
conds = [("none", 0.0, None)] + [(n, al, v * al) for n, v in (("d+", dp), ("d-", dm)) for al in ALPHAS] + [("random", 2.0, dr * 2.0)]
A = np.full((len(test), len(conds)), -1)
S = np.full((len(test), len(conds), 2), np.nan, np.float32)
for j, t in enumerate(test[:LIMIT]):
    for c, (_, _, v) in enumerate(conds):
        A[j, c], S[j, c] = generate(its[idx[t]], v)
    if j % 40 == 0:
        print(a.model, j, len(test), flush=True)
os.makedirs(D, exist_ok=True)
np.savez(out, ans=A, scores=S, conds=np.array([c[0] for c in conds]), alpha=np.array([c[1] for c in conds]), test=test, layer=l, nL=nL,
         norm_dp=float(dp.norm()), norm_dm=float(dm.norm()), norm_mu0=float(mu0.norm()), cos_pm=float(torch.nn.functional.cosine_similarity(dp, dm, 0)),
         img_pos_norm=float(H.norm(dim=1).mean()))
print(a.model, "steer", {} if LIMIT < 10 ** 9 else {f"{c[0]}@{c[1]:g}": round(float(np.mean(A[:, k][A[:, k] >= 0] == 1)), 3) for k, c in enumerate(conds)},
      "cos(d+, d-)", round(float(torch.nn.functional.cosine_similarity(dp, dm, 0)), 3), flush=True)
