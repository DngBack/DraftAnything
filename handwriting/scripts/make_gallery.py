"""Extractor audit gallery (plan §4.1): 50 lines from distinct writers and 200 word crops,
annotated with core band (blue = top, red = baseline), slant line (green) and stroke width."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def draw_word(ax, r, dx=0, dy=0):
    x0, y0, x1, y1 = r.box_x0 - dx, r.box_y0 - dy, r.box_x1 - dx, r.box_y1 - dy
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="0.6", lw=0.4))
    if r.v_baseline:
        b = r.baseline_y - dy
        ax.plot([x0, x1], [b, b], color="red", lw=0.8)
        ax.plot([x0, x1], [b - r.core_h, b - r.core_h], color="#2a6fdb", lw=0.8)
    if r.v_slant:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        h = (y1 - y0) / 2
        t = np.tan(np.deg2rad(r.slant))
        ax.plot([cx - h * t, cx + h * t], [cy + h, cy - h], color="green", lw=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/gallery"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv("data/manifest.csv", dtype={"writer_id": str}).set_index("line_id")
    w = pd.concat([pd.read_csv(f"data/words_{s}.csv.gz", dtype={"writer_id": str}) for s in ["train", "val1"]])
    for c in ["v_baseline", "v_core", "v_slant", "v_slant_grad", "v_stroke"]:
        w[c] = w[c].fillna(False).astype(bool)
    rng = np.random.default_rng(a.seed)
    lines = (w.drop_duplicates("line_id").groupby("writer_id").line_id
             .apply(lambda s: s.iloc[rng.integers(len(s))]))
    lines = rng.permutation(lines.to_numpy())[:50]
    for page in range(5):
        fig, axes = plt.subplots(10, 1, figsize=(16, 22))
        for ax, lid in zip(axes, lines[page * 10:(page + 1) * 10]):
            img = np.asarray(Image.open(man.loc[lid, "image_path"]).convert("L"))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            for r in w[w.line_id == lid].itertuples():
                if r.n_ink > 0:
                    draw_word(ax, r)
            ax.set_title(f"{lid}  writer {man.loc[lid, 'writer_id']}", fontsize=8, loc="left")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(a.out / f"lines_{page}.png", dpi=80)
        plt.close(fig)

    cand = w[(w.n_ink > 0) & w.text.astype(str).str.len().ge(2)]
    cand = cand.iloc[rng.permutation(len(cand))[:200]]
    for page in range(4):
        fig, axes = plt.subplots(10, 5, figsize=(16, 20))
        for ax, r in zip(axes.ravel(), cand.iloc[page * 50:(page + 1) * 50].itertuples()):
            img = np.asarray(Image.open(man.loc[r.line_id, "image_path"]).convert("L"))
            pad = 4
            y0, y1 = max(int(r.box_y0) - pad, 0), int(r.box_y1) + pad
            x0, x1 = max(int(r.box_x0) - pad, 0), int(r.box_x1) + pad
            ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
            draw_word(ax, r, dx=x0, dy=y0)
            sl = f"{r.slant:.0f}°" if r.v_slant else "slant–"
            sw = f"sw {r.stroke_w:.1f}" if r.v_stroke else "sw–"
            ax.set_title(f"{r.text!s} | {sl} | {sw} | core {'ok' if r.v_core else '–'}", fontsize=7)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(a.out / f"words_{page}.png", dpi=80)
        plt.close(fig)
    cand[["line_id", "pos", "text", "slant", "stroke_w", "core_h", "v_slant", "v_core", "v_stroke"]] \
        .to_csv(a.out / "word_sample.csv", index=False)


if __name__ == "__main__":
    main()
