"""Overlay sheets for the visual audit (doc §6): divisions and non-division track ends."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import tifffile

from .ctc import Sequence


def _crop(img, cy, cx, r):
    y0, x0 = int(max(cy - r, 0)), int(max(cx - r, 0))
    return img[y0:y0 + 2 * r, x0:x0 + 2 * r], y0, x0


def _panel(ax, seq: Sequence, t, ids, cy, cx, r, title):
    img = tifffile.imread(seq.images()[t])
    tra = tifffile.imread(seq.markers()[t])
    im, y0, x0 = _crop(img, cy, cx, r)
    lab = tra[y0:y0 + 2 * r, x0:x0 + 2 * r]
    ax.imshow(im, cmap="gray")
    ax.contour(lab > 0, [0.5], colors="yellow", linewidths=0.4)
    for i, c in zip(ids, ["red", "cyan", "lime"]):
        if (lab == i).any():
            ax.contour(lab == i, [0.5], colors=c, linewidths=1.0)
    ax.set_title(title, fontsize=6)
    ax.axis("off")


def division_sheet(seq, tracks, obs, out, n=20, r=40, seed=0):
    """n random divisions: parent at its last frame | children at their first frame."""
    rng = np.random.default_rng(seed)
    kids = tracks[tracks.parent_id != 0].groupby("parent_id").gt_track_id.apply(list)
    kids = kids[kids.map(len) >= 1]
    pick = rng.choice(kids.index.to_numpy(), size=min(n, len(kids)), replace=False)
    pos = obs.set_index(["gt_track_id", "frame_index"])[["center_y", "center_x"]]
    T = tracks.set_index("gt_track_id")
    fig, axes = plt.subplots(4, 10, figsize=(15, 6.5))
    for k, p in enumerate(pick):
        e = T.at[p, "end_frame"]
        cy, cx = pos.loc[(p, e)]
        c = kids[p]
        b = T.at[c[0], "start_frame"]
        _panel(axes.flat[2 * k], seq, e, [p], cy, cx, r, f"P{p} t{e}")
        _panel(axes.flat[2 * k + 1], seq, b, c, cy, cx, r, f"{'+'.join(map(str, c))} t{b}")
    for ax in axes.flat[2 * len(pick):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def end_sheet(seq, tracks, obs, out, reasons=("near_border", "unknown_end"), n=20, r=40, seed=0):
    """n random non-division, non-video-end track ends: last frame | next frame."""
    rng = np.random.default_rng(seed)
    cand = tracks[tracks.end_reason.isin(reasons)]
    if cand.empty:  # nothing to inspect; do not leave a blank sheet behind
        Path(out).unlink(missing_ok=True)
        return
    pick = cand.sample(min(n, len(cand)), random_state=int(rng.integers(1 << 31)))
    pos = obs.set_index(["gt_track_id", "frame_index"])[["center_y", "center_x"]]
    last = max(seq.markers())
    fig, axes = plt.subplots(4, 10, figsize=(15, 6.5))
    for k, row in enumerate(pick.itertuples()):
        e = row.end_frame
        cy, cx = pos.loc[(row.gt_track_id, e)]
        _panel(axes.flat[2 * k], seq, e, [row.gt_track_id], cy, cx, r,
               f"{row.gt_track_id} t{e} {row.end_reason}")
        _panel(axes.flat[2 * k + 1], seq, min(e + 1, last), [], cy, cx, r, f"t{min(e + 1, last)}")
    for ax in axes.flat[2 * len(pick):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
