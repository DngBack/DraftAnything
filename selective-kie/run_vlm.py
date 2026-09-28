"""Frozen VLM pass over CORD: (1) JSON extraction with token logprobs, (2) teacher-forced scores of every
on-page candidate value (+ null) for every field. Usage: python run_vlm.py SPLIT [MODEL] [LIMIT]"""
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import FIELDS, candidates, load_cord

split = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-VL-4B-Instruct"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
OUT = f"outputs/{MODEL.split('/')[-1]}/{split}" + (f".shard{os.environ['SHARD'][0]}" if os.environ.get("SHARD") else "") + ".jsonl"
MAX_SIDE = 1280  # ~1.1k visual tokens for CORD images

EXTRACT_Q = (
    "Extract the following fields from this receipt. Reply with only a JSON object with exactly these keys:\n"
    + "\n".join(f'- "{f}": {d}' for f, (_, d) in FIELDS.items())
    + "\nCopy each value exactly as printed on the receipt. Use null if the field is not printed."
)
FIELD_Q = "What is the {} on this receipt? Reply with only the value exactly as printed, or null if it is not printed."

proc = AutoProcessor.from_pretrained(MODEL)
proc.tokenizer.padding_side = "left"
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
END = "<|im_end|>"


def chat(q):
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False)


def shrink(img):
    s = MAX_SIDE / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s))) if s < 1 else img


@torch.no_grad()
def extract(imgs):
    """Greedy JSON extraction; returns (text, token ids, raw-logit logprobs) per doc."""
    inp = proc(text=[chat(EXTRACT_Q)] * len(imgs), images=imgs, padding=True, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=256, do_sample=False, output_logits=True, return_dict_in_generate=True)
    gen = out.sequences[:, inp.input_ids.shape[1]:]
    lp = torch.stack([l.float().log_softmax(-1) for l in out.logits], 1).gather(-1, gen[..., None])[..., 0]
    res = []
    for ids, l in zip(gen.tolist(), lp.tolist()):
        n = next((i for i, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids(END))), len(ids))
        res.append((tok.decode(ids[:n]), ids[:n], l[:n]))
    return res


KEY_RE = {f: re.compile(r'"%s"\s*:\s*(null|"([^"]*)"|-?[\d.,]+)' % f) for f in FIELDS}


def parse(text, ids, lps):
    """Per field: predicted string (None=null/missing) and logprobs of tokens overlapping the value span."""
    ends = [len(tok.decode(ids[: i + 1])) for i in range(len(ids))]  # char end offset of each token
    starts = [0] + ends[:-1]
    out = {}
    for f, rx in KEY_RE.items():
        m = rx.search(text)
        if not m:
            out[f] = {"pred": None, "lps": [], "missing_key": True}
            continue
        g = 2 if m.group(2) is not None else 1
        a, b = m.span(g)
        if a == b:  # empty string "" -> treat as null, score the quote tokens
            a, b = a - 1, b + 1
        pred = None if m.group(1) == "null" or m.group(g) == "" else m.group(g)
        out[f] = {"pred": pred, "lps": [lp for s, e, lp in zip(starts, ends, lps) if s < b and e > a]}
    return out


@torch.no_grad()
def score_candidates(img, cands, bs=40):
    """log P(answer + <|im_end|> | image, field question) for each field x (candidates + null)."""
    answers = [c["surface"] for c in cands] + ["null"]
    reqs = []
    for f, (_, d) in FIELDS.items():
        pre = chat(FIELD_Q.format(d))
        n_pre = len(tok(pre).input_ids)
        for a in answers:
            ids = tok(pre + a + END).input_ids
            assert ids[:n_pre] == tok(pre).input_ids
            reqs.append((f, a, pre + a + END, len(ids) - n_pre))
    scores = []
    for i in range(0, len(reqs), bs):
        chunk = reqs[i: i + bs]
        k = max(r[3] for r in chunk)
        inp = proc(text=[r[2] for r in chunk], images=[img] * len(chunk), padding=True, return_tensors="pt").to("cuda")
        logits = model(**inp, logits_to_keep=k + 1).logits[:, :-1].float().log_softmax(-1)  # predicts last k tokens
        tgt = inp.input_ids[:, -k:]
        tl = logits.gather(-1, tgt[..., None])[..., 0]
        for j, r in enumerate(chunk):
            scores.append(tl[j, k - r[3]:].sum().item())
    res = {f: {"cands": [], "null": None} for f in FIELDS}
    for (f, a, _, _), s in zip(reqs, scores):
        if a == "null" and len(res[f]["cands"]) == len(cands):
            res[f]["null"] = s
        else:
            res[f]["cands"].append(s)
    return res


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    docs = load_cord(split)[:LIMIT]
    if os.environ.get("SHARD"):  # "i/n": process every n-th doc starting at i (parallel GPUs)
        i, n = map(int, os.environ["SHARD"].split("/"))
        docs = docs[i::n]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT)}
    docs = [d for d in docs if d["id"] not in done]
    t0 = time.time()
    with open(OUT, "a") as fo:
        for i in range(0, len(docs), 8):
            batch = docs[i: i + 8]
            imgs = [shrink(d["image"]) for d in batch]
            for d, img, (text, ids, lps) in zip(batch, imgs, extract(imgs)):
                cands = candidates(d["lines"])
                rec = {"id": d["id"], "raw": text, "fields": parse(text, ids, lps),
                       "cands": [c["value"] for c in cands], "surfaces": [c["surface"] for c in cands],
                       "scores": score_candidates(img, cands)}
                fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(f"{split} {i + len(batch)}/{len(docs)} {time.time() - t0:.0f}s", flush=True)
