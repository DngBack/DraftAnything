"""Side-by-side sheets for eyeballing H2: the same val1 line as real / Emuru / TS-GAN, stacked.

Only lines with CER <= 0.2 in all three sources (same set as the H2 tables). All images are
resized to the same height so baselines and spacing are comparable by eye.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SRC = [("real", "data/dyn/cer_real_val1.csv"), ("emuru", "data/dyn/cer_emuru_val1.csv"),
       ("tsgan", "data/dyn/cer_tsgan_val1.csv")]
FIX = [("real", "data/dyn/cer_real_val1.csv"), ("tsgan", "data/dyn/cer_tsgan_val1.csv"),
       ("tsgan_donor", "data/dyn/fix/cer_tsgan_donor.csv"), ("tsgan_style", "data/dyn/fix/cer_tsgan_style.csv"),
       ("emuru", "data/dyn/cer_emuru_val1.csv"), ("emuru_donor", "data/dyn/fix/cer_emuru_donor.csv"),
       ("emuru_style", "data/dyn/fix/cer_emuru_style.csv")]
LABEL = {"real": "REAL (IAM)", "emuru": "Emuru", "tsgan": "TS-GAN", "tsgan_donor": "TS-GAN+donor",
         "tsgan_style": "TS-GAN+style", "emuru_donor": "Emuru+donor", "emuru_style": "Emuru+style"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/compare"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--h", type=int, default=96)
    ap.add_argument("--fix", action="store_true", help="raw vs re-layout fix variants")
    a = ap.parse_args()
    srcs = FIX if a.fix else SRC
    tabs = {k: pd.read_csv(p).set_index("line_id") for k, p in srcs}
    ok = set.intersection(*[set(t.index[t.cer <= 0.2]) for k, t in tabs.items() if k in LABEL and "_" not in k])
    ok = ok.intersection(*[set(t.index) for t in tabs.values()])
    ids = sorted(np.random.default_rng(0).choice(sorted(ok), a.n, replace=False))
    a.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for lid in ids:
        ims = []
        for k, _ in srcs:
            im = Image.open(tabs[k].loc[lid, "image_path"]).convert("L")
            # Fixed lines grow taller with drift: scale them like their 64 px raw output, not to a.h.
            f = a.h / (64 if "_" in k else im.height)
            ims.append((k, im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)))
        w = max(im.width for _, im in ims) + 120
        ys = np.cumsum([24] + [im.height + 12 for _, im in ims])
        sheet = Image.new("L", (w, int(ys[-1])), 255)
        d = ImageDraw.Draw(sheet)
        d.text((6, 4), f"{lid}  |  {tabs['real'].loc[lid, 'transcript']}", fill=0, font=font)
        for (k, im), y in zip(ims, ys):
            d.text((6, y + im.height // 2 - 6), LABEL[k], fill=0, font=font)
            sheet.paste(im, (120, int(y)))
            d.line([(0, y + im.height + 5), (w, y + im.height + 5)], fill=200)
        sheet.save(a.out / f"{lid}.png")
    print(f"{len(ids)} sheets -> {a.out}")


if __name__ == "__main__":
    main()
