"""H2 audit: TS-GAN (Davis et al., BMVC 2020), a line-level GAN trained on IAM lines.

Its own form split (data/sets.json @ 081bb8e) puts only our train/unassigned forms in its
train set; our val1/val2 forms are in its "valid" set, so val1 targets are unseen in training.
Style is extracted from another line of the same writer (same rule as Emuru/B0), the text is
the target's written transcript. Follows the repo's own 'from-image' path: 64 px, 1 - x/128.
Run from anywhere; the repo is put on sys.path and used as cwd for its relative paths.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from hwline.segment import written_tokens  # noqa: E402

REPO = Path("/home/jovyan/bachdx2/repos/ts-gan")
CKPT = Path("/home/jovyan/bachdx2/repos/ts-gan-weights/IAM_weights/"
            "IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG/"
            "checkpoint-iteration175000.pth")


def pick_style_line(man, line):
    same = man[(man.writer_id == line.writer_id) & (man.line_id != line.line_id)]
    other = same[same.form_id != line.form_id]
    pool = other if len(other) else same
    if not len(pool):
        return None, False
    return pool.loc[pool.transcript.str.len().idxmax()], len(other) > 0


def load_model(gpu):
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import model as M  # noqa: F401  (registers the architectures used by eval below)
    from model import HWWithStyle  # noqa: F401
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    for k in [k for k in ck["state_dict"] if "style_from_normal" in k]:
        del ck["state_dict"][k]
    cfg = ck["config"]
    for k in cfg:
        if "pretrained" in k:
            cfg[k] = None
    cfg["model"]["RUN"] = True
    net = eval(cfg["arch"], vars(M))(cfg["model"])
    net.load_state_dict(ck["state_dict"])
    net.eval().to(gpu)
    net.count_std = 0; net.dup_std = 0
    char_set = json.load(open(cfg["data_loader"]["char_file"]))
    return net, char_set["char_to_idx"]


def prep(path, gpu):
    im = cv2.imread(str(path), 0)
    im = cv2.resize(im, (0, 0), fx=64 / im.shape[0], fy=64 / im.shape[0], interpolation=cv2.INTER_CUBIC)
    x = 1.0 - im[None].astype(np.float32) / 128.0
    return torch.from_numpy(x).to(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out.resolve()
    man = pd.read_csv(HERE / "data/manifest.csv", dtype={"writer_id": str})
    man = man[man.quality_flag == "ok"]
    tgt = man[man.split == a.split]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gpu = torch.device("cuda")
    net, c2i = load_model(gpu)
    from utils import string_utils
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for line in tgt.itertuples():
        sty, other = pick_style_line(man, line)
        if sty is None:
            continue
        text = " ".join(written_tokens(line.transcript))
        if any(c not in c2i for c in text):
            skipped += 1
            continue
        p = out / f"{line.line_id}.png"
        if not p.exists():
            with torch.no_grad():
                style = net.extract_style(prep(sty.image_path, gpu)[None], None, 1)
                lab = torch.from_numpy(string_utils.str2label_single(text, c2i).astype(np.int64))
                lab = lab[:, None].to(gpu)
                im = net(lab, torch.IntTensor([lab.size(0)]), style)
            cv2.imwrite(str(p), ((1 - im[0].permute(1, 2, 0)) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8))
        rows.append({"image_path": str(p), "transcript": text, "writer_id": line.writer_id,
                     "line_id": line.line_id, "style_line": sty.line_id, "style_other_form": other})
    pd.DataFrame(rows).to_csv(out / "list_0.csv", index=False)
    print(json.dumps({"generated": len(rows), "skipped_charset": skipped}))


if __name__ == "__main__":
    main()
