"""H2 audit: Emuru (CVPR 2025, zero-shot, trained on synthetic fonts -> no IAM writer leakage).

For each target line, the style is another line of the same writer (other form preferred;
same choice rule as B0), given with its transcript as style text; gen text is the target's
written transcript. IAM background above the line threshold is set to white (Emuru was
trained on clean renders). Output is 64 px high; the profile is measured after resizing to
the same height as the real reference.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as F

from hwline.segment import written_tokens

# Pinned HF revision; its modeling code was read before use (no I/O beyond from_pretrained).
EMURU_REV = "c75b78dd6ea0306123ad5a12b18b9ae4a927e149"


def pick_style_line(man, line):
    same = man[(man.writer_id == line.writer_id) & (man.line_id != line.line_id)]
    other = same[same.form_id != line.form_id]
    pool = other if len(other) else same
    if not len(pool):
        return None, False
    # Deterministic: the longest line in the pool (most style evidence).
    return pool.loc[pool.transcript.str.len().idxmax()], len(other) > 0


def load_style(rec):
    g = np.asarray(Image.open(rec.image_path).convert("L")).copy()
    g[g >= rec.threshold] = 255
    im = Image.fromarray(g).convert("RGB")
    im = im.resize((max(im.width * 64 // im.height, 8), 64), Image.BILINEAR)
    return F.normalize(F.to_tensor(im), [0.5], [0.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModel
    torch.manual_seed(a.seed)
    model = AutoModel.from_pretrained("blowing-up-groundhogs/emuru", revision=EMURU_REV,
                                      trust_remote_code=True).cuda().eval()
    man = pd.read_csv("data/manifest.csv", dtype={"writer_id": str})
    man = man[man.quality_flag == "ok"]
    tgt = man[man.split == a.split].iloc[a.shard::a.n_shards]
    a.out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for line in tgt.itertuples():
        p = a.out / f"{line.line_id}.png"
        sty, other = pick_style_line(man, line)
        if sty is None:
            continue
        style_text = " ".join(written_tokens(sty.transcript))
        gen_text = " ".join(written_tokens(line.transcript))
        empty = a.out / f"{line.line_id}.empty"
        if not p.exists() and not empty.exists():
            img = load_style(sty).cuda()
            px_per_char = img.shape[-1] / max(len(style_text), 1)
            max_tok = int(len(gen_text) * px_per_char * 1.6 / 8) + 32
            with torch.inference_mode():
                out = model.generate(style_text=style_text, gen_text=gen_text, style_img=img,
                                     max_new_tokens=max_tok)
            # Emuru sometimes stops at once and returns a zero-width image: a generation
            # failure, kept as a marker so the audit can report it.
            if out.width == 0 or out.height == 0:
                empty.touch()
            else:
                out.convert("L").save(p)
        if empty.exists():
            failed.append(line.line_id)
            continue
        rows.append({"image_path": str(p), "transcript": gen_text, "writer_id": line.writer_id,
                     "line_id": line.line_id, "style_line": sty.line_id, "style_other_form": other})
    pd.DataFrame(rows).to_csv(a.out / f"list_{a.shard}.csv", index=False)
    print(json.dumps({"shard": a.shard, "generated": len(rows), "empty": len(failed)}))


if __name__ == "__main__":
    main()
