"""B0 (plan §5): SpiS-GAN word generator, words concatenated into a line.

For each target line (val1), the style comes from one word image of the same writer taken
from a different line (different form when possible). Each written token is generated at
32 px with the public IAM checkpoint and pasted left-to-right with a fixed 16 px gap.
  --noise line : one noise vector per line (fixed style + fixed noise)
  --noise word : fresh noise per word (iid word-level variation)
Caveat: the checkpoint was trained on a 339-writer split; val1 writers may be seen writers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

SPIS = Path("/home/jovyan/bachdx2/repos/spis-gan")
sys.path.insert(0, str(SPIS))
from lib.alphabet import Alphabets, strLabelConverter  # noqa: E402
from lib.utils import yaml2config  # noqa: E402
from networks.BigGAN_networks import Generator  # noqa: E402
from networks.module32 import SharedBackbone, StyleEncoder  # noqa: E402

from hwline.segment import written_tokens  # noqa: E402


def load_models(ckpt, device):
    cfg = yaml2config(str(SPIS / "configs/SpiS_gan_iam_32.yml"))
    G = Generator(**cfg.GenModel).to(device)
    E = StyleEncoder(**cfg.EncModel).to(device)
    S = SharedBackbone(**cfg.SharedBackbone).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    for m in (G, E, S):
        m.load_state_dict(sd[type(m).__name__])
        m.eval()
    return cfg, G, E, S


def pick_reference(man, line):
    """Longest alphabetic word of the same writer from another line, other form preferred."""
    same = man[(man.writer_id == line.writer_id) & (man.line_id != line.line_id)]
    other_form = same[same.form_id != line.form_id]
    pool = other_form if len(other_form) else same
    best = None
    for r in pool.itertuples():
        for t, b in zip(json.loads(r.word_texts), json.loads(r.word_boxes)):
            if b and t.isalpha() and (best is None or len(t) > len(best[1])):
                best = (r, t, b)
    return best, len(other_form) > 0


def ref_tensor(rec, box, device):
    g = np.asarray(Image.open(rec.image_path).convert("L"))
    x0, y0, x1, y1 = box
    crop = Image.fromarray(g[y0:y1, x0:x1])
    w = max(int(round(crop.width * 32 / crop.height)), 16)
    a = np.asarray(crop.resize((w, 32), Image.BILINEAR)).astype(np.float32) / 255.0
    return torch.from_numpy(a * 2 - 1)[None, None].to(device), torch.tensor([w], device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/jovyan/bachdx2/datasets/spis_gan/bestIAM.pth")
    ap.add_argument("--split", default="val1")
    ap.add_argument("--noise", choices=["line", "word"], default="line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gap", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    cfg, G, E, S = load_models(a.ckpt, dev)
    conv = strLabelConverter("all")
    alpha = set(Alphabets["all"])
    man = pd.read_csv("data/manifest.csv", dtype={"writer_id": str})
    man = man[man.quality_flag == "ok"]
    tgt = man[man.split == a.split]
    a.out.mkdir(parents=True, exist_ok=True)
    rows, skipped, same_form = [], 0, 0
    noise_dim = cfg.GenModel.style_dim - cfg.EncModel.style_dim
    with torch.no_grad():
        for line in tgt.itertuples():
            toks = written_tokens(line.transcript)
            if any(c not in alpha for t in toks for c in t) or any(len(t) > 20 for t in toks):
                skipped += 1
                continue
            ref, other = pick_reference(man, line)
            if ref is None:
                skipped += 1
                continue
            same_form += not other
            img, ln = ref_tensor(ref[0], ref[2], dev)
            style = E(img, ln, S)
            n = len(toks)
            noise = torch.randn(1 if a.noise == "line" else n, noise_dim, device=dev).expand(n, -1)
            z = torch.cat([noise, style.expand(n, -1)], 1)
            lbs, lens = conv.encode(toks) if n > 1 else conv.encode([toks[0], toks[0]])
            lbs, lens = torch.as_tensor(lbs)[:n].long().to(dev), torch.as_tensor(lens)[:n].long().to(dev)
            fake = G(z, lbs, lens)
            pieces = []
            for i in range(n):
                w = int(lens[i]) * cfg.char_width
                pieces.append(((fake[i, 0, :, :w].clamp(-1, 1) + 1) * 127.5).cpu().numpy())
                if i < n - 1:
                    pieces.append(np.full((32, a.gap), 255.0))
            arr = np.concatenate(pieces, 1).astype(np.uint8)
            p = a.out / f"{line.line_id}.png"
            Image.fromarray(arr).save(p)
            rows.append({"image_path": str(p), "transcript": " ".join(toks), "writer_id": line.writer_id,
                         "line_id": line.line_id, "ref_line": ref[0].line_id, "ref_word": ref[1]})
    pd.DataFrame(rows).to_csv(a.out / "list.csv", index=False)
    print(json.dumps({"generated": len(rows), "skipped": skipped, "ref_same_form": same_form}))


if __name__ == "__main__":
    main()
