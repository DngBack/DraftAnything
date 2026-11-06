"""Protocol D outputs (docs/protocol_decide.md): one row per cell x stimulus x prompt. Resumable, shardable."""
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
from evc.data import DATA  # noqa: E402
from evc.degrade import SCALES, degrade  # noqa: E402
from evc.stimuli import best_font, blur_canvas, draw_box, edit, restore, swap_digit  # noqa: E402
from evc.vlm import load_vlm  # noqa: E402
from run_percept import PROMPTS as P0  # noqa: E402

ADDRESS = 'In the table, find the row labeled "{row}" and the column labeled "{col}". '
PROMPTS = dict(
    copy=P0["copy"],
    copyu=P0["copyu"],
    addr=P0["addr"],
    copyu2=ADDRESS + "Output only the number in that cell, exactly as printed. A wrong number is much worse than saying you cannot read it: "
                     "unless you are certain of every digit, answer UNREADABLE.",
    conf=ADDRESS + 'Output the number in that cell exactly as printed, then " | ", then your confidence from 0 to 100 that you read every digit correctly.',
    boxcopy=ADDRESS + "The cell is outlined in red. Output only the number in that cell, exactly as printed.",
    boxonly="Output only the number inside the red box, exactly as printed.",
    program=P0["program"],
)
B1_LEVELS = ("s100", "s50", "s40", "s30")
B2_LEVELS = ("s50", "s40", "s30")


@torch.no_grad()
def gen(vlm, img, text, max_new=40):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    t = vlm.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = vlm.proc(text=[t], images=[img], return_tensors="pt")
    dt = getattr(vlm, "dtype", None)
    inp = inp.to(vlm.device, dtype=dt) if dt is not None else inp.to(vlm.device)
    out = vlm.model.generate(**inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, top_k=None, top_p=None, temperature=None,
                             return_dict_in_generate=True, output_scores=True)
    seq = out.sequences[0, inp["input_ids"].shape[1]:]
    toks, lps, mar = [], [], []
    for s, tok in zip(out.scores, seq.tolist()):
        if tok in vlm.eos:
            continue
        lp = torch.log_softmax(s[0].float(), -1)
        top = lp.topk(2).values
        toks.append(vlm.tok.decode([tok]))
        lps.append(round(lp[tok].item(), 4))
        mar.append(round((top[0] - top[1]).item(), 4))
    return dict(text=vlm.tok.decode(seq, skip_special_tokens=True).strip(), lp=sum(lps) / len(lps) if lps else 0.0,
                lpmin=min(lps) if lps else 0.0, toks=toks, lps=lps, mar=mar)


def stimuli(it):
    """Yield (block, level, cond, prompt, image). Edited pages are built once per cell."""
    img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
    font = best_font(img, it["word"], it["bbox"])
    pages = dict(orig=img, null=edit(img, it["word"], it["bbox"], it["word"], font),
                 cf=edit(img, it["word"], it["bbox"], swap_digit(it["word"], it["pos"], it["d_s"]), font))
    plan = dict(orig=("copy", "addr", "copyu", "copyu2", "conf", "program"), null=("copy",), cf=("copy", "addr"))
    for cond, prompts in plan.items():
        for lv in B1_LEVELS:
            x = degrade(pages[cond], lv, it["uid"])
            for p in prompts:
                yield 1, lv, cond, p, x
    for lv in B2_LEVELS:
        canvas = blur_canvas(img, SCALES[lv])
        g = restore(canvas, img, it["num_bboxes"])
        yield 2, lv, "base", "copy", canvas
        yield 2, lv, "G", "copy", g
        yield 2, lv, "L", "boxcopy", draw_box(canvas, it["bbox"])
        yield 2, lv, "LG", "boxcopy", draw_box(g, it["bbox"])
    yield 3, "s100", "decoy", "boxonly", draw_box(img, it["decoy_bbox"])
    yield 3, "s40", "decoy", "boxonly", draw_box(blur_canvas(img, SCALES["s40"]), it["decoy_bbox"])
    yield 3, "s100", "L", "boxcopy", draw_box(img, it["bbox"])
    # box on the wrong cell while the prompt still names the target's labels: does the cue hijack the read?
    yield 3, "s100", "decoy", "boxcopy", draw_box(img, it["decoy_bbox"])
    yield 3, "s40", "decoy", "boxcopy", draw_box(blur_canvas(img, SCALES["s40"]), it["decoy_bbox"])
    # nothing to read at all: floor for refusal and stated confidence
    blank = Image.new("RGB", img.size, "white")
    for p in ("copy", "copyu2", "conf"):
        yield 3, "s100", "blank", p, blank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="decide_set.json")
    ap.add_argument("--name", default="decide")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n_sh = map(int, a.shard.split("/"))
    items = json.load(open(DATA / a.set))[: a.limit or None][k::n_sh]
    out = ROOT / "results" / f"{a.name}_{a.tag}_{k}.jsonl"
    done = set()
    if out.exists():
        done = {tuple(json.loads(l)[x] for x in ("qid", "block", "level", "cond", "prompt")) for l in open(out)}
    vlm = load_vlm(a.model)
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        if "num_bboxes" not in it:  # pilot items used for the smoke test
            from evc.closure import YEAR_RE
            from evc.data import is_num, load_doc, page_words

            words, _ = page_words(load_doc(it["uid"], "train"))
            nums = [w for w in words if is_num(w["text"]) and w["text"] != it["word"]]
            it = dict(it, num_bboxes=[w["bbox"] for w in words if is_num(w["text"]) and (w["text"] == it["word"] or not YEAR_RE.match(w["text"]))], decoy_word=nums[0]["text"], decoy_bbox=nums[0]["bbox"])
        for block, lv, cond, p, img in stimuli(it):
            if (it["qid"], block, lv, cond, p) in done:
                continue
            r = gen(vlm, img, PROMPTS[p].format(row=it["row"], col=it["col"], q=it["question"]))
            r.update(qid=it["qid"], uid=it["uid"], block=block, level=lv, cond=cond, prompt=p, img_wh=list(img.size))
            f.write(json.dumps(r) + "\n")
            f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
