"""E9 (dev): recovery rate of targeted repair actions, per failure type.

Base = E8 outputs (direct, program). Actions:
  scale_ocr : numbers in the model's expression -> matching OCR words -> heuristic closure unit caption
              (+ "percentage"-type questions -> percent).                         0 VLM calls
  scale_vlm : ask the VLM which scale applies to the numbers it used.              1 call
  zoom      : crop the table band whose lines share tokens with the question, upscale x2,
              re-run the program prompt on the crop.                                1 call
  verify    : show the proposed expression, ask the model to check and output a corrected one.  1 call
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
from evc.closure import SCALE_RE, layout, relations  # noqa: E402
from evc.data import DATA, is_num, load_doc, page_words  # noqa: E402
from evc.learned import q_tokens, norm  # noqa: E402
from evc.metrics import parse_nums  # noqa: E402
from evc.vlm import load_vlm  # noqa: E402
from run_e8 import PROGRAM, gen  # noqa: E402

SCALE_Q = ("Which scale applies to the numbers needed to answer the question: thousand, million, billion, percent, "
           "or none? Answer with one word.\nQuestion: {q}")
VERIFY = ("Question: {q}\nA proposed arithmetic expression for the answer is: {e}\n"
          "Check it against the document: are the numbers copied from the right rows and columns, and is the "
          "operation right for the question (e.g. change = new - old, percentage change = (new - old) / old, "
          "average = sum / count)? Output only the corrected expression (or the same one if it is correct).")
PCT_Q = re.compile(r"percent|percentage|%|proportion|ratio|rate|margin", re.I)


def canon_scale(t: str) -> str:
    t = t.lower()
    for s in ("thousand", "million", "billion", "percent"):
        if s[:5] in t or (s == "percent" and "%" in t):
            return s
    return ""


def scale_from_ocr(words, page_w, nums, question):
    if PCT_Q.search(question):
        return "percent"
    votes = []
    for n in nums:
        for i, w in enumerate(words):
            if not is_num(w["text"]):
                continue
            wn = parse_nums(w["text"])
            if wn and abs(wn[0] - abs(n)) < 1e-6 * max(1, abs(n)):
                u = relations(words, i, page_w)["unit"]
                txt = " ".join(words[j]["text"] for j in u)
                s = canon_scale(txt) if SCALE_RE.search(txt) or "%" in txt else ""
                votes.append(s)
    votes = [v for v in votes if v]
    return max(set(votes), key=votes.count) if votes else ""


def zoom_crop(img, words, page_w, question, pad=60):
    """Band of lines sharing tokens with the question (+ header lines above), full width, upscaled x2."""
    lines, line_of, para = layout(words, page_w)
    qt = q_tokens(question)
    hit = [li for li, ln in enumerate(lines) if li not in para and len({norm(words[i]["text"]) for i in ln} & qt) >= 1
           and any(is_num(words[i]["text"]) for i in ln)]
    if not hit:
        return None
    ys = [words[i]["bbox"][1] for li in hit for i in lines[li]] + [words[i]["bbox"][3] for li in hit for i in lines[li]]
    y0, y1 = max(0, min(ys) - 260), min(img.height, max(ys) + pad)
    crop = img.crop((0, y0, img.width, y1))
    return crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "hf" / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--base", default="e8")
    ap.add_argument("--prefix", default="e9")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    base = [json.loads(l) for l in open(ROOT / "results" / f"{args.base}_{args.tag}.jsonl")][: args.limit or None]
    out = ROOT / "results" / f"{args.prefix}_{args.tag}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    vlm = load_vlm(args.model)
    f = open(out, "a")
    t0 = time.time()
    for n, r in enumerate(base):
        if r["qid"] in done:
            continue
        words, (W, H) = page_words(load_doc(r["uid"], args.split))
        img = Image.open(DATA / "pages" / f"{r['uid']}.png").convert("RGB")
        q = r["question"]
        a = dict(qid=r["qid"])
        a["scale_ocr"] = scale_from_ocr(words, W, parse_nums(r["program"]), q)
        a["scale_vlm"] = canon_scale(gen(vlm, img, SCALE_Q.format(q=q), 8))
        a["scale_direct"] = canon_scale(r["direct"])
        crop = zoom_crop(img, words, W, q)
        a["zoom"] = gen(vlm, crop, PROGRAM.format(q=q), 64) if crop is not None else None
        a["verify"] = gen(vlm, img, VERIFY.format(q=q, e=r["program"]), 64)
        f.write(json.dumps(a) + "\n")
        f.flush()
        print(f"[{n + 1}/{len(base)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
