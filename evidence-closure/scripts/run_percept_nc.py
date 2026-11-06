"""Pilot C (docs/protocol_percept_nc.md): beam hypotheses for the copy prompt on original pages."""
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
from evc.degrade import degrade  # noqa: E402
from evc.vlm import load_vlm  # noqa: E402
from run_percept import PROMPTS  # noqa: E402

K = 8


@torch.no_grad()
def beam(vlm, img, text):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    t = vlm.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = vlm.proc(text=[t], images=[img], return_tensors="pt").to(vlm.device)
    out = vlm.model.generate(**inp, max_new_tokens=16, do_sample=False, num_beams=K, num_return_sequences=K, repetition_penalty=1.0,
                             top_k=None, top_p=None, temperature=None, return_dict_in_generate=True, output_scores=True)
    n = inp["input_ids"].shape[1]
    return [vlm.tok.decode(s[n:], skip_special_tokens=True).strip() for s in out.sequences], out.sequences_scores.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    items = json.load(open(DATA / "percept_set.json"))
    out = ROOT / "results" / f"percept_nc_{a.tag}.jsonl"
    done = {(json.loads(l)["qid"], json.loads(l)["level"]) for l in open(out)} if out.exists() else set()
    vlm = load_vlm(a.model)
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        img0 = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
        for lv in ("s50", "s40", "s30"):
            if (it["qid"], lv) in done:
                continue
            texts, scores = beam(vlm, degrade(img0, lv, it["uid"]), PROMPTS["copy"].format(row=it["row"], col=it["col"]))
            f.write(json.dumps(dict(qid=it["qid"], uid=it["uid"], level=lv, texts=texts, scores=scores)) + "\n")
            f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
