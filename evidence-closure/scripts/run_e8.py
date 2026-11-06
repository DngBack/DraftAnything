"""E8 (dev, diagnostic): where do full-page DocVQA errors come from — reading, calculation or reasoning?

Per question, full page, greedy, no repetition penalty:
  direct  : the standard answer prompt (same as E1 `full`)
  extract : list the numbers needed, with row / column labels, no computation  -> read recall vs gold facts
  program : one arithmetic expression over document numbers, executed in Python -> separates calc vs reasoning
  hint    : direct prompt + "The relevant values are: <gold facts>"           -> evidence-complete ceiling
  hintprog: program prompt + gold facts                                         -> reasoning with complete evidence
"""
import argparse
import ast
import json
import operator
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evc.data import DATA  # noqa: E402
from evc.metrics import parse_nums, scale_correct, value_correct  # noqa: E402
from evc.vlm import PROMPT, load_vlm  # noqa: E402

EXTRACT = ("List every number in the document that is needed to answer the question. For each number write one line: "
           "value | row label | column label. Do not compute the answer.\nQuestion: {q}")
PROGRAM = ("Write ONE arithmetic expression that computes the answer, using numbers copied exactly from the document "
           "(operators + - * / and parentheses only, no units, no commas inside numbers). If the answer is not a "
           "calculation, write the answer text instead. Output only the expression.\nQuestion: {q}")
HINT = "The relevant values in the document are: {f}.\n"

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str):
    """Evaluate + - * / () over numeric literals; None if not a pure arithmetic expression."""
    e = expr.strip().strip("`").split("=")[0].replace("[", "(").replace("]", ")")
    e = re.sub(r"(?<=\d),(?=\d{3})", "", e).replace("$", "").replace("%", "")
    e = re.sub(r"\((\d[\d.]*)\)", r"(-\1)", e) if re.fullmatch(r"\(\d[\d.]*\)", e.strip()) else e
    try:
        node = ast.parse(e, mode="eval").body
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError

    try:
        return ev(node)
    except (ValueError, ZeroDivisionError):
        return None


def read_recall(text: str, facts) -> float:
    got = parse_nums(text)
    hits = 0
    for f in facts:
        fn = parse_nums(str(f))
        if fn and any(abs(g - fn[0]) <= 1e-6 * max(1, abs(fn[0])) for g in got):
            hits += 1
    return hits / max(1, len(facts))


@torch.no_grad()
def gen(vlm, img, text, max_new):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    t = vlm.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = vlm.proc(text=[t], images=[img], return_tensors="pt").to(vlm.device)
    out = vlm.model.generate(**inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0,
                             top_k=None, top_p=None, temperature=None)
    return vlm.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "hf" / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--set", default="e1_set.json")
    ap.add_argument("--qfile", default="tatdqa_dataset_dev.json")
    ap.add_argument("--prefix", default="e8")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    gold = {q["uid"]: q for e in json.load(open(DATA / args.qfile)) for q in e["questions"]}
    items = json.load(open(DATA / args.set))[: args.limit or None]
    out = ROOT / "results" / f"{args.prefix}_{args.tag}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    vlm = load_vlm(args.model)
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        if it["qid"] in done:
            continue
        g = gold[it["qid"]]
        img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
        q = it["question"]
        facts = g["facts"]
        hint = HINT.format(f=", ".join(str(x) for x in facts))
        r = dict(qid=it["qid"], uid=it["uid"], answer_type=it["answer_type"], scale=it["scale"], n_facts=it["n_facts"],
                 answer=it["answer"], facts=facts, derivation=g["derivation"], question=q)
        r["direct"] = gen(vlm, img, PROMPT.format(q=q), 64)
        r["extract"] = gen(vlm, img, EXTRACT.format(q=q), 160)
        r["program"] = gen(vlm, img, PROGRAM.format(q=q), 64)
        r["hint"] = gen(vlm, img, hint + PROMPT.format(q=q), 64)
        r["hintprog"] = gen(vlm, img, hint + PROGRAM.format(q=q), 64)
        r["direct_v"] = value_correct(r["direct"], it["answer"])
        r["direct_s"] = scale_correct(r["direct"], it["scale"])
        r["hint_v"] = value_correct(r["hint"], it["answer"])
        r["read_recall"] = read_recall(r["extract"], facts)
        for k in ("program", "hintprog"):
            val = safe_eval(r[k])
            r[f"{k}_val"] = val
            r[f"{k}_v"] = value_correct(f"{val}", it["answer"]) if val is not None else value_correct(r[k], it["answer"])
        gv = safe_eval(g["derivation"]) if g["derivation"] else None
        r["gold_deriv_v"] = value_correct(f"{gv}", it["answer"]) if gv is not None else None  # scorer sanity
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min direct={r['direct_v']} read={r['read_recall']:.2f} "
              f"prog={r['program_v']} hint={r['hint_v']}", flush=True)


if __name__ == "__main__":
    main()
