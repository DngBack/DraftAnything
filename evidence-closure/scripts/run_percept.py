"""Pilot B outputs (docs/protocol_percept_cf.md): one row per question x level x edit x prompt."""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.data import DATA  # noqa: E402
from evc.degrade import degrade  # noqa: E402
from evc.stimuli import best_font, edit, swap_digit  # noqa: E402
from evc.vlm import load_vlm  # noqa: E402
from run_e8 import PROGRAM  # noqa: E402
from run_h3 import gen_lp  # noqa: E402

LEVELS = ("s100", "s60", "s50", "s40", "s30")
ADDRESS = 'In the table, find the row labeled "{row}" and the column labeled "{col}". '
PROMPTS = {
    "copy": ADDRESS + "Output only the number in that cell, exactly as printed.",
    "copyu": ADDRESS + 'Output only the number in that cell, exactly as printed. If its digits are not legible, answer UNREADABLE.',
    "addr": ADDRESS + "Write ONE arithmetic expression that multiplies the number in that cell by 2, copying the number exactly "
                      "as printed. Output only the expression.",
    "program": PROGRAM,
}
PLAN = {"orig": ("copy", "copyu", "addr", "program"), "null": ("copy",), "cf": ("copy", "addr", "program")}


def stimulus(it, cond):
    img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
    if cond == "orig":
        return img
    font = best_font(img, it["word"], it["bbox"])
    text = it["word"] if cond == "null" else swap_digit(it["word"], it["pos"], it["d_s"])
    return edit(img, it["word"], it["bbox"], text, font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = json.load(open(DATA / "percept_set.json"))[: a.limit or None]
    out = ROOT / "results" / f"percept_{a.tag}.jsonl"
    done = set()
    if out.exists():
        done = {tuple(json.loads(l)[k] for k in ("qid", "level", "cond", "prompt")) for l in open(out)}
    vlm = load_vlm(a.model)
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        for cond, prompts in PLAN.items():
            base = stimulus(it, cond)
            for lv in LEVELS:
                img = degrade(base, lv, it["uid"])
                for p in prompts:
                    if (it["qid"], lv, cond, p) in done:
                        continue
                    text = PROMPTS[p].format(row=it["row"], col=it["col"], q=it["question"])
                    txt, lp, lpmin = gen_lp(vlm, img, text, 48)
                    f.write(json.dumps(dict(qid=it["qid"], uid=it["uid"], level=lv, cond=cond, prompt=p, text=txt, lp=lp, lpmin=lpmin)) + "\n")
                    f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
