"""Interventional attribution data. For each question at one image-quality level:

  direct, hint (always); program, hintprog (arithmetic only: computation cannot be repaired elsewhere);
  with --op: opdirect, opprog, opboth (arithmetic with a parseable gold derivation; P1O1C1 needs no model).
Gold facts/derivation are the intervention itself, not features of any policy.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.data import DATA  # noqa: E402
from evc.degrade import degrade  # noqa: E402
from evc.vlm import PROMPT, load_vlm  # noqa: E402
from run_e8 import HINT, PROGRAM, gen  # noqa: E402

OP = "The answer is computed with this formula, where each letter stands for a number in the document: {t}.\n"
NUM = re.compile(r"\$?\d[\d,]*\.?\d*%?")


def template(deriv):
    """Mask the numbers of a gold derivation: '(5,1-4.7)/4.7' -> '(a-b)/b'. None if it is not a plain formula."""
    d = (deriv or "").strip()
    if not d or "##" in d or ">" in d or "<" in d or "=" in d or re.search(r"[A-Za-z]", d):
        return None
    names, out, pos = {}, [], 0
    for m in NUM.finditer(d):
        out.append(d[pos:m.start()])
        k = m.group(0)
        names.setdefault(k, "abcdefghijklmnop"[min(len(names), 15)])
        out.append(names[k])
        pos = m.end()
    out.append(d[pos:])
    return "".join(out).strip() if names else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "hf" / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--level", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="drop the first questions (calibration)")
    ap.add_argument("--op", action="store_true")
    ap.add_argument("--name", default="attr")
    a = ap.parse_args()
    gold = {q["uid"]: q for e in json.load(open(DATA / "tatdqa_dataset_train.json")) for q in e["questions"]}
    items = json.load(open(DATA / "attr_set.json"))[a.skip: a.limit or None]
    out = ROOT / "results" / f"{a.name}_{a.tag}_{a.level}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    vlm = load_vlm(a.model)
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        if it["qid"] in done:
            continue
        g = gold[it["qid"]]
        img = degrade(Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB"), a.level, it["uid"])
        q = it["question"]
        hint = HINT.format(f=", ".join(str(x) for x in g["facts"]))
        r = dict(qid=it["qid"], uid=it["uid"], answer_type=it["answer_type"], scale=it["scale"], answer=it["answer"],
                 n_facts=it["n_facts"], facts=g["facts"], derivation=g["derivation"], question=q, level=a.level,
                 img_wh=list(img.size))
        r["direct"] = gen(vlm, img, PROMPT.format(q=q), 64)
        r["hint"] = gen(vlm, img, hint + PROMPT.format(q=q), 64)
        if it["answer_type"] == "arithmetic":
            r["program"] = gen(vlm, img, PROGRAM.format(q=q), 64)
            r["hintprog"] = gen(vlm, img, hint + PROGRAM.format(q=q), 64)
            t = template(g["derivation"])
            r["op_template"] = t
            if a.op and t:
                o = OP.format(t=t)
                r["opdirect"] = gen(vlm, img, o + PROMPT.format(q=q), 64)
                r["opprog"] = gen(vlm, img, o + PROGRAM.format(q=q), 64)
                r["opboth"] = gen(vlm, img, hint + o + PROMPT.format(q=q), 64)
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
