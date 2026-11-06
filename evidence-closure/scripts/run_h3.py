"""H3 data: run EVERY action on every question (docs/protocol_h3.md), so policies differ only in selection.

Saved per question: outputs of direct / program / zoom / verify / scale_vlm, scale_ocr, mean token
log-prob and min token log-prob of direct and program, and the gold fields needed for scoring.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.data import DATA, load_doc, page_words  # noqa: E402
from evc.metrics import parse_nums  # noqa: E402
from evc.vlm import PROMPT, load_vlm  # noqa: E402
from run_e8 import PROGRAM  # noqa: E402
from run_e9 import SCALE_Q, VERIFY, canon_scale, scale_from_ocr, zoom_crop  # noqa: E402


@torch.no_grad()
def gen_lp(vlm, img, text, max_new):
    """Greedy decode; returns (text, mean token log-prob, min token log-prob)."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    t = vlm.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = vlm.proc(text=[t], images=[img], return_tensors="pt").to(vlm.device)
    out = vlm.model.generate(**inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, top_k=None,
                             top_p=None, temperature=None, return_dict_in_generate=True, output_scores=True)
    seq = out.sequences[0, inp["input_ids"].shape[1]:]
    lps = [torch.log_softmax(s[0].float(), -1)[tok].item() for s, tok in zip(out.scores, seq)]
    keep = [lp for lp, tok in zip(lps, seq.tolist()) if tok not in vlm.eos]
    txt = vlm.tok.decode(seq, skip_special_tokens=True).strip()
    return txt, (sum(keep) / len(keep) if keep else 0.0), (min(keep) if keep else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "hf" / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--qfile", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="", help="output name (default: split), e.g. test2")
    args = ap.parse_args()
    gold = {q["uid"]: q for e in json.load(open(DATA / args.qfile)) for q in e["questions"]}
    items = json.load(open(DATA / args.set))[: args.limit or None]
    out = ROOT / "results" / f"h3_{args.name or args.split}_{args.tag}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    vlm = load_vlm(args.model)
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        if it["qid"] in done:
            continue
        g = gold[it["qid"]]
        words, (W, H) = page_words(load_doc(it["uid"], args.split))
        img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
        q = it["question"]
        r = dict(qid=it["qid"], uid=it["uid"], answer_type=it["answer_type"], scale=it["scale"],
                 n_facts=it["n_facts"], answer=it["answer"], facts=g["facts"], derivation=g["derivation"], question=q)
        r["direct"], r["direct_lp"], r["direct_lpmin"] = gen_lp(vlm, img, PROMPT.format(q=q), 64)
        r["program"], r["program_lp"], r["program_lpmin"] = gen_lp(vlm, img, PROGRAM.format(q=q), 64)
        r["scale_vlm"] = canon_scale(gen_lp(vlm, img, SCALE_Q.format(q=q), 8)[0])
        r["scale_direct"] = canon_scale(r["direct"])
        r["scale_ocr"] = scale_from_ocr(words, W, parse_nums(r["program"]), q)
        crop = zoom_crop(img, words, W, q)
        r["zoom"] = gen_lp(vlm, crop, PROGRAM.format(q=q), 64)[0] if crop is not None else None
        r["verify"] = gen_lp(vlm, img, VERIFY.format(q=q, e=r["program"]), 64)[0]
        # grounding: fraction of numbers in each expression that occur as an OCR word on the page
        page_nums = {round(abs(x), 6) for w in words for x in parse_nums(w["text"])}
        for k in ("program", "zoom", "verify"):
            ns = [round(abs(x), 6) for x in parse_nums(r[k] or "")]
            r[f"{k}_ground"] = (sum(x in page_nums for x in ns) / len(ns)) if ns else 0.0
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
