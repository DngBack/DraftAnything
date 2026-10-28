"""G7 T (docs/g7_protocol.md): the comparison rule in text. Tests T1-T2 (analysis in g7_psy.py).

Every illusynP item is given to the model as text only: the two compared lengths as integer pixels and the item's
question. The assistant turn is opened with "{" (" {" after a template ending in a non-space) and the decision is the higher of the Yes / No first-token scores
(the G3 decision step without the free-form prefix).
Output: results/g7/text_<model>.npz with y, ans, scores, key, fam, regime, diff, L (pixel lengths).

usage: python g7_text.py --model qwen3vl8b
"""
import argparse, os, numpy as np, torch
from gen_calib import items
from pilot import Runner, OUT

D = f"{OUT}/g7"
NAMES = {"mullerlyer": ("the top horizontal line", "the bottom horizontal line"),
         "ponzo": ("the upper horizontal bar", "the lower horizontal bar"),
         "verticalhorizontal": ("the horizontal line", "the vertical line")}


def prompt(it):
    a, b = NAMES[it["group"]]
    la, lb = (int(round(v)) for v in it["L"])
    return f"In an image, {a} is {la} pixels long and {b} is {lb} pixels long. {it['q']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    out = f"{D}/text_{a.model}.npz"
    os.makedirs(D, exist_ok=True)
    if not os.path.exists(out):
        R = Runner(a.model, ["Yes", "No"])
        tok = R.p.tokenizer
        cand = [sorted({i for p in ("", " ") for v in (c, c.lower()) for i in tok.encode(p + v, add_special_tokens=False)
                        if tok.decode([i]).strip().lower() == c.lower()}) for c in ("Yes", "No")]  # as diag_gen.cand_ids
        its = items("illusynP")
        sc = []
        with torch.no_grad():
            for it in its:
                msgs = [{"role": "user", "content": [{"type": "text", "text": prompt(it)}]}]
                s = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                s += "{" if s[-1].isspace() else " {"  # e.g. LLaVA's "ASSISTANT:" would merge ":{" into one token
                bos = tok.bos_token is not None and s.startswith(tok.bos_token)
                ids = tok(s, add_special_tokens=not bos, return_tensors="pt").to("cuda")
                lg = R.m(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"]).logits[0, -1].float()
                sc.append(torch.stack([lg[c].max() for c in cand]).cpu().numpy())
        sc = np.stack(sc)
        y = np.array([it["cands"].index(it["y"]) for it in its])
        np.savez(out, y=y, ans=sc.argmax(1), scores=sc, key=np.array([it["key"] for it in its]),
                 fam=np.array([it["group"] for it in its]), regime=np.array([it["regime"] for it in its]),
                 diff=np.array([it["diff"] for it in its]), L=np.array([it["L"] for it in its]))
        print(f"{a.model} text: example prompt: {s!r}", flush=True)
        print(f"{a.model} text: n={len(y)} saved", flush=True)
