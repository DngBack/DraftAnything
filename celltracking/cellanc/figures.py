"""Explanatory figures for the research report (outputs/figures/fig*.png).

Reads outputs/{index,benchmarks,results,tables} and raw images under data/raw. Nothing here
feeds back into the benchmark; these are views for humans.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

from .baselines import BASELINES, expansion_compensate, mutual_nearest_siblings
from .benchmark import cell_cycle_frames
from .loader import Windows
from .stats import BASELINE_STYLE, GRID, INK, MUTED, ORDER, SERIES, _sched_key, _style

NAMES = {"one_in_a_million": "OIAM (C. glutamicum)", "sim_plus": "SIM+ (Fluo-N2DH-SIM+)",
         "hsc": "HSC (BF-C2DL-HSC)", "musc": "MuSC (BF-C2DL-MuSC)", "hela": "HeLa (PhC-C2DL-HeLa)"}
SEQ_DIR = {"M0": ("one_in_a_million", "00"), "M4": ("one_in_a_million", "04"),
           "HeLa-02": ("hela", "02"), "MuSC-02": ("musc", "02"), "HSC-02": ("hsc", "02"),
           "SIM-02": ("sim_plus", "02"), "MuSC-01": ("musc", "01")}
RED, LIME = "#d62728", "#1baf7a"
DPI = 150


# ---------- helpers ----------

def _norm(img, lo=1, hi=99.5):
    img = img.astype(np.float32)
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / max(b - a, 1e-6), 0, 1)


def _boundary(mask):
    return mask != ndimage.grey_erosion(mask, size=3)


def _hue_colors(ids, seed=0):
    """Distinct-ish colours for many labels: golden-ratio hues in random order."""
    ids = list(ids)
    order = np.random.default_rng(seed).permutation(len(ids))
    return {i: matplotlib.colors.hsv_to_rgb(((k * 0.618034) % 1, 0.75, 0.95))
            for i, k in zip(ids, order)}


def _overlay(gray, mask, colors, alpha=0.5, outline=None, outline_color=(1, 0, 0), dim_rest=None):
    """RGB image: gray + filled colours for labels in `colors`; optional outline for label set."""
    rgb = np.repeat(gray[..., None], 3, -1)
    n = int(mask.max()) + 1
    lut = np.zeros((n, 3), np.float32)
    has = np.zeros(n, bool)
    for k, c in colors.items():
        if k < n:
            lut[k], has[k] = c, True
    sel = has[mask]
    rgb[sel] = (1 - alpha) * rgb[sel] + alpha * lut[mask][sel]
    if dim_rest is not None:
        rest = (mask > 0) & ~sel
        b = _boundary(mask) & rest
        rgb[b] = dim_rest
    if outline:
        m = np.isin(mask, list(outline))
        b = _boundary(np.where(m, mask, 0)) & m
        b = ndimage.binary_dilation(b)
        rgb[b] = outline_color
    return rgb


class Seq:
    """Evaluator-side view of one indexed sequence (for figures only)."""

    def __init__(self, out: Path, name: str, sid: str):
        src = out / "index" / name / sid
        self.name, self.sid = name, sid
        self.frames = pd.read_parquet(src / "frames.parquet").set_index("frame_index")
        self.tr = pd.read_parquet(src / "tracks.parquet")
        self.obs = pd.read_parquet(src / "observations.parquet")
        self.tracks = {r.gt_track_id: (r.start_frame, r.end_frame, r.parent_id)
                       for r in self.tr.itertuples()}
        self.kids = defaultdict(list)
        for L, (_, _, p) in self.tracks.items():
            if p:
                self.kids[p].append(L)

    def img(self, t):
        return tifffile.imread(self.frames.at[t, "image_path"])

    def mask(self, t):
        return tifffile.imread(self.frames.at[t, "marker_path"])

    def pos(self, t):
        o = self.obs[self.obs.frame_index == t]
        return dict(zip(o.gt_track_id, zip(o.center_y, o.center_x)))


def _crop(arr, cy, cx, r):
    H, W = arr.shape[:2]
    y0, x0 = int(np.clip(cy - r, 0, max(H - 2 * r, 0))), int(np.clip(cx - r, 0, max(W - 2 * r, 0)))
    return arr[y0:y0 + 2 * r, x0:x0 + 2 * r], (y0, x0)


def _noaxis(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _scalebar(ax, um_per_px, um, shape, color="white"):
    h, w = shape[:2]
    L = um / um_per_px
    ax.plot([w - L - 0.05 * w, w - 0.05 * w], [h * 0.93] * 2, color=color, lw=3,
            solid_capstyle="butt")
    ax.text(w - L / 2 - 0.05 * w, h * 0.9, f"{um:g} µm", color=color, ha="center", va="bottom",
            fontsize=7)


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path.name, flush=True)


# ---------- dataset figures ----------

def fig_gallery(out, cfgs, path):
    """Full frame + zoom per dataset, GT markers outlined."""
    picks = [("one_in_a_million", "00", 560, 90), ("sim_plus", "01", None, 80),
             ("hsc", "02", None, 60), ("musc", "02", None, 90), ("hela", "02", None, 60)]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), gridspec_kw={"height_ratios": [1.25, 1]})
    for j, (name, sid, t, r) in enumerate(picks):
        s = Seq(out, name, sid)
        counts = s.obs.groupby("frame_index").size()
        if t is None:  # frame at 80% of the movie that has markers
            t = int(counts.index[int(0.8 * (len(counts) - 1))])
        g, m = _norm(s.img(t)), s.mask(t)
        rgb = np.repeat(g[..., None], 3, -1)
        rgb[ndimage.binary_dilation(_boundary(m), iterations=1 if name == "one_in_a_million" else 2)] = (1, 0.8, 0)
        # zoom on the densest part
        p = s.obs[s.obs.frame_index == t]
        cy, cx = p.center_y.median(), p.center_x.median()
        crop, (y0, x0) = _crop(rgb, cy, cx, r)
        ax = axes[0, j]
        ax.imshow(rgb, interpolation="lanczos")
        ax.add_patch(mpatches.Rectangle((x0, y0), 2 * r, 2 * r, fill=False, ec="#00e5ff", lw=1.2))
        _scalebar(ax, cfgs[name]["pixel_size_um"],
                  {"one_in_a_million": 10, "sim_plus": 20}.get(name, 100), rgb.shape)
        cfg = cfgs[name]
        ax.set_title(f"{NAMES[name]}\n{cfg['time_step_min']:g} min/frame · {cfg['pixel_size_um']:g} "
                     f"µm/px · t={t} · {counts.get(t, 0)} cells", fontsize=8, color=INK)
        _noaxis(ax)
        axes[1, j].imshow(crop, interpolation="nearest")
        _noaxis(axes[1, j])
        axes[1, j].set_title("zoom (cyan box)", fontsize=7, color=MUTED)
    fig.suptitle("Figure: the five datasets, one representative frame each; yellow = GT TRA marker "
                 "contours", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_counts(out, cfgs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axes[0]
    _style(ax)
    for i, sid in enumerate(["00", "01", "02", "03", "04"]):
        o = pd.read_parquet(out / "index/one_in_a_million" / sid / "observations.parquet")
        c = o.groupby("frame_index").size()
        ax.plot(c.index, c.values, color=SERIES[i], lw=1.4, label=f"M{i}")
    ax.set_yscale("log")
    ax.set_xlabel("frame (= minute)", fontsize=8, color=MUTED)
    ax.set_ylabel("annotated cells in frame", fontsize=8, color=MUTED)
    ax.axvspan(700, 800, color=GRID, alpha=0.6, lw=0)
    ax.text(705, 3, "chamber full,\ncells pushed out", fontsize=7, color=MUTED)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper left")
    ax.set_title("OIAM: exponential growth from 1–4 founders to ~1,400 cells", fontsize=9,
                 color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    for i, name in enumerate(ORDER[1:]):
        for sid, ls in [("01", "-"), ("02", "--")]:
            o = pd.read_parquet(out / "index" / name / sid / "observations.parquet")
            c = o.groupby("frame_index").size()
            ax.plot(c.index * cfgs[name]["time_step_min"] / 60, c.values, color=SERIES[i + 1],
                    ls=ls, lw=1.3, label=f"{name} {sid}")
    ax.set_yscale("log")
    ax.set_xlabel("time (hours)", fontsize=8, color=MUTED)
    ax.set_title("CTC sets (solid = 01, dashed = 02)", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def fig_cycles(out, cfgs, path):
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.8))
    for ax, (i, name) in zip(axes, enumerate(ORDER)):
        _style(ax)
        dur = np.concatenate([cell_cycle_frames(Seq(out, name, sid).tracks)
                              for sid in sorted(p.name for p in (out / "index" / name).iterdir())])
        dur = dur * cfgs[name]["time_step_min"]
        ax.hist(dur, bins=30, color=SERIES[i], alpha=0.85)
        med = cfgs[name]["cell_cycle_median_frames"] * cfgs[name]["time_step_min"]
        ax.axvline(med, color=INK, lw=1, ls="--")
        ax.set_title(f"{NAMES[name]}\ndashed = horizon median (train/adapt) {med:g} min; all "
                     f"movies n = {len(dur)}",
                     fontsize=8, color=INK)
        ax.set_xlabel("division-to-division time (min)", fontsize=7, color=MUTED)
    axes[0].set_ylabel("tracks", fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, path)


def fig_quality(out, path):
    """Data-quality examples found in the audit."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    s = Seq(out, "one_in_a_million", "00")
    t = 760
    ends = s.tr[(s.tr.end_frame > t) & (s.tr.end_frame <= t + 10) & (s.tr.start_frame <= t)]
    cols = {**{L: matplotlib.colors.to_rgb(SERIES[1]) for L in
               ends[ends.end_reason == "near_border"].gt_track_id},
            **{L: matplotlib.colors.to_rgb(SERIES[2]) for L in
               ends[ends.end_reason == "unknown_end"].gt_track_id},
            **{L: matplotlib.colors.to_rgb(SERIES[0]) for L in
               ends[ends.end_reason == "division"].gt_track_id}}
    rgb = _overlay(_norm(s.img(t)), s.mask(t), cols, alpha=0.75)
    axes[0].imshow(rgb)
    H, W = rgb.shape[:2]
    axes[0].add_patch(mpatches.Rectangle((30, 30), W - 60, H - 60, fill=False, ec="white",
                                         ls="--", lw=1))
    axes[0].set_title(f"M0 t={t}: tracks ending in the next 10 frames\nblue = divides, orange = "
                      f"near border (≤30 px), green = unknown", fontsize=8, color=INK)
    c, _ = _crop(rgb, H / 2, 0, 110)
    axes[1].imshow(c)
    axes[1].set_title("zoom on the left edge: cells pushed out of the chamber", fontsize=8,
                      color=INK)
    for ax in axes[:2]:
        _noaxis(ax)
    ax = axes[2]
    _style(ax)
    tr = pd.concat(pd.read_parquet(out / "index/one_in_a_million" / sid / "tracks.parquet")
                   for sid in ["00", "01", "02", "03", "04"])
    tr = tr[tr.end_frame < 799]
    for i, (reason, g) in enumerate(tr.groupby("end_reason")):
        ax.hist(g.end_frame, bins=np.arange(0, 801, 20), histtype="step", lw=1.6,
                color=SERIES[i], label=f"{reason} ({len(g)})")
    ax.set_yscale("log")
    ax.set_xlabel("track end frame", fontsize=8, color=MUTED)
    ax.set_title("OIAM M0–M4: why tracks end before the movie ends", fontsize=8, color=INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, path)


# ---------- task / benchmark figures ----------

def _window(out, alias, a, b, sid="ends"):
    wid = f"{alias}_a{a:04d}_b{b:04d}_{sid}"
    root = out / "benchmarks/v0"
    tg = pd.read_parquet(root / "labels/targets" / f"{alias}.parquet")
    gm = pd.read_parquet(root / "labels/gt_mapping" / f"{alias}.parquet")
    to_gt = dict(zip(zip(gm.frame_index, gm.local_detection_id), gm.gt_track_id))
    return wid, tg[tg.window_id == wid], to_gt


def _lineage_colors(tg, to_gt, a, b):
    """gt anchor id -> colour; gt target id -> gt anchor id (valid, in-FOI)."""
    ok = tg[(tg.target_status == "valid_anchor") & tg.in_foi]
    anc = {to_gt[(b, int(t))]: to_gt[(a, int(p))] for t, p in
           zip(ok.target_detection_id, ok.ancestor_detection_id)}
    return anc, _hue_colors(sorted(set(anc.values())))


def fig_task(out, path, alias="M4", a=406, b=538):
    name, sid = SEQ_DIR[alias]
    s = Seq(out, name, sid)
    wid, tg, to_gt = _window(out, alias, a, b)
    anc, col = _lineage_colors(tg, to_gt, a, b)
    pb = s.obs[s.obs.frame_index == b]
    cy, cx = pb.center_y.median(), pb.center_x.median()
    r = int(max(pb.center_y.max() - pb.center_y.min(), pb.center_x.max() - pb.center_x.min()) / 2) + 40
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    ma, mb = s.mask(a), s.mask(b)
    ra = _overlay(_norm(s.img(a)), ma, col, alpha=0.6)
    rb = _overlay(_norm(s.img(b)), mb, {t: col[p] for t, p in anc.items()}, alpha=0.6)
    for ax, im, title in [(axes[0], ra, f"frame a = {a}: {ma.max() and len(np.unique(ma)) - 1} "
                                         f"cells, each anchor a colour"),
                          (axes[1], rb, f"frame b = {b} (+{b - a} min = 2 cycles): cell coloured "
                                        f"by its true ancestor at a")]:
        c, _ = _crop(im, cy, cx, r)
        ax.imshow(c)
        _noaxis(ax)
        ax.set_title(title, fontsize=8, color=INK)
    _scalebar(axes[0], 0.072, 5, (2 * r, 2 * r))
    # family sizes
    ax = axes[2]
    _style(ax)
    sizes = pd.Series(anc).value_counts()
    vc = sizes.value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=SERIES[0])
    ax.set_xlabel("descendants at b per anchor", fontsize=8, color=MUTED)
    ax.set_ylabel("anchors", fontsize=8, color=MUTED)
    ax.set_title(f"window {wid}\n{len(anc)} targets from {len(sizes)} anchors "
                 f"(mean {sizes.mean():.2f} descendants)", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def _descendants(s, root, a, b):
    """Tracks of the lineage of `root` that overlap [a, b]."""
    out, stack = [], [root]
    while stack:
        L = stack.pop()
        st, en, _ = s.tracks[L]
        if st > b:
            continue
        out.append(L)
        stack.extend(s.kids[L])
    return out


def fig_lineage(out, path, alias="M4", a=406, b=538):
    name, sid = SEQ_DIR[alias]
    s = Seq(out, name, sid)
    wid, tg, to_gt = _window(out, alias, a, b)
    anc, _ = _lineage_colors(tg, to_gt, a, b)
    fam = pd.Series(anc).value_counts()
    pa = s.pos(a)
    ca = np.mean(list(pa.values()), axis=0)
    # a 4-descendant family near the colony centre, fully binary
    cand = [k for k, v in fam.items() if v == 4]
    root = min(cand, key=lambda k: np.hypot(*(np.array(pa[k]) - ca)))
    lin = _descendants(s, root, a, b)
    depth = {root: 0}
    for L in sorted(lin, key=lambda L: s.tracks[L][0]):
        for k in s.kids[L]:
            depth[k] = depth[L] + 1
    # tree layout: leaves in order, parent at mean of children
    ypos = {}

    def place(L, nxt=[0]):
        ks = [k for k in s.kids[L] if k in lin]
        if not ks:
            ypos[L] = nxt[0]
            nxt[0] += 1
        else:
            for k in ks:
                place(k)
            ypos[L] = np.mean([ypos[k] for k in ks])
    place(root)
    cols = _hue_colors(lin, seed=3)
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    for L in lin:
        st, en, p = s.tracks[L]
        x0, x1 = max(st, a), min(en, b)
        ax.plot([x0, x1], [ypos[L]] * 2, color=cols[L], lw=4, solid_capstyle="butt")
        ax.text(x0 + 1, ypos[L] + 0.12, f"track {L}", fontsize=7, color=MUTED)
        if L != root and p in ypos:
            ax.plot([st - 0.5] * 2, [ypos[p], ypos[L]], color=MUTED, lw=1)
            ax.plot(st - 0.5, ypos[p], "o", color=INK, ms=5)
    snaps = np.linspace(a, b, 5).round().astype(int)
    for t in snaps:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_xlim(a - 2, b + 8)
    ax.set_yticks([])
    ax.set_xlabel("frame (min)", fontsize=8, color=MUTED)
    ax.set_title(f"{alias}: lineage of anchor track {root} between a={a} and b={b}; black dots = "
                 f"divisions (depth at b = {max(depth[L] for L in lin)}); grey lines = snapshots "
                 f"below", fontsize=9, color=INK, loc="left")
    for j, t in enumerate(snaps):
        axi = fig.add_subplot(gs[1, j])
        live = [L for L in lin if s.tracks[L][0] <= t <= s.tracks[L][1]]
        p = s.pos(t)
        cy, cx = np.mean([p[L] for L in live if L in p], axis=0)
        m = s.mask(t)
        rgb = _overlay(_norm(s.img(t)), m, {L: cols[L] for L in live}, alpha=0.65,
                       dim_rest=(0.9, 0.8, 0.2))
        c, _ = _crop(rgb, cy, cx, 70)
        axi.imshow(c)
        _noaxis(axi)
        axi.set_title(f"t = {t} (+{t - a} min): {len(live)} cell(s)", fontsize=8, color=INK)
    fig.tight_layout()
    _save(fig, path)
    return root


def fig_schedules(out, cfgs, path, alias="M4", a=406, b=538):
    """Which frames each Experiment-A schedule keeps, and what fraction of targets then has a
    division hidden between two kept frames."""
    root = out / "benchmarks/v0"
    W = [json.loads(l) for l in open(root / "inputs/windows" / f"{alias}.jsonl")]
    W = {w["schedule_id"]: w for w in W if w["anchor_frame"] == a and w["target_frame"] == b}
    tg = pd.read_parquet(root / "labels/targets" / f"{alias}.parquet")
    order = sorted(W, key=_sched_key)
    name, sid = SEQ_DIR[alias]
    s = Seq(out, name, sid)
    kids = defaultdict(int)
    for L, (_, _, p) in s.tracks.items():
        kids[p] += 1
    divs = [s.tracks[L][1] for L in s.tracks if kids[L] >= 2 and a <= s.tracks[L][1] < b]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.hist(divs, bins=np.arange(a, b + 1, 2), bottom=-1.9, color=GRID,
            weights=np.full(len(divs), 1.6 / max(np.histogram(divs, np.arange(a, b + 1, 2))[0].max(), 1)))
    for i, sidk in enumerate(order):
        fr = W[sidk]["observed_frames"]
        ax.plot(fr, [i] * len(fr), "|", color=SERIES[0] if sidk.startswith("s") else SERIES[1],
                ms=10 if len(fr) < 40 else 6, mew=1.4)
        t = tg[(tg.window_id == W[sidk]["window_id"]) & (tg.target_status == "valid_anchor")]
        pct = 100 * (t.hidden_intermediates > 0).mean()
        ax.text(b + 3, i, f"{len(fr):>3} frames · {pct:4.1f}% targets with hidden division",
                fontsize=7.5, va="center", color=INK)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlim(a - 3, b + 60)
    ax.set_ylim(-2.1, len(order))
    ax.text(a, -1.2, f"division events between a and b ({len(divs)}), histogram", fontsize=7,
            color=MUTED)
    ax.set_xlabel("frame (min)", fontsize=8, color=MUTED)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(f"Experiment A on one (a, b) = ({a}, {b}) of {alias}: every schedule keeps a "
                 f"and b; blue = regular stride, orange = endpoints / random", fontsize=9,
                 color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def fig_strata(out, cfgs, path):
    st = pd.read_csv(out / "tables/strata.csv")
    hid = pd.read_csv(out / "tables/hidden.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    _style(ax)
    rows = []
    for name in ORDER:
        d = st[st.dataset == name]
        for H in sorted(d.horizon_frames.unique()):
            g = d[d.horizon_frames == H]
            w = g.targets
            rows.append((f"{name}\nH={H / cfgs[name]['cell_cycle_median_frames']:.1f}c",
                         [np.average(g[f"depth{k}{'+' if k == 3 else ''}_%"].fillna(0), weights=w)
                          for k in range(4)]))
    lab = [r[0] for r in rows]
    v = np.array([r[1] for r in rows])
    left = np.zeros(len(rows))
    for k in range(4):
        ax.barh(range(len(rows)), v[:, k], left=left, color=SERIES[k],
                label=f"depth {k}{'+' if k == 3 else ''}")
        left += v[:, k]
    ax.set_yticks(range(len(rows)), lab, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("% of valid targets", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_title("Generations between ancestor and target (c = cell cycles)", fontsize=9,
                 color=INK, loc="left", pad=22)
    ax = axes[1]
    _style(ax)
    h = hid[hid.schedule_id.str.startswith("s")]
    ax.plot(h.schedule_id.str[1:].astype(int), h.pct_targets_with_hidden, color=SERIES[0],
            marker="o", lw=2)
    for sidk in ["ends", "rand1", "rand3"]:
        r = hid[hid.schedule_id == sidk]
        if len(r):
            ax.axhline(r.pct_targets_with_hidden.iat[0], color=SERIES[1], lw=1, ls=":")
            ax.text(1, r.pct_targets_with_hidden.iat[0] + 1.5, sidk, fontsize=7, color=SERIES[1])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("stride between kept frames (min)", fontsize=8, color=MUTED)
    ax.set_ylabel("% targets with a hidden division", fontsize=8, color=MUTED)
    ax.set_title("OIAM, H = 132 min: divisions that fall\nbetween two kept frames", fontsize=9,
                 color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def fig_baseline_schematic(path):
    """Synthetic examples: why expansion compensation and capacity-2 assignment help."""
    from .baselines import _lap
    rng = np.random.default_rng(4)
    n = 14
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = np.sqrt(rng.uniform(0, 1, n)) * 10
    A = np.c_[np.arange(1, n + 1), rad * np.sin(ang), rad * np.cos(ang)]
    kids, par = [], []
    for i, (_, y, x) in enumerate(A):
        th = rng.uniform(0, np.pi)
        drift = rng.normal(0, 1.2, 2)  # each family also moves on its own
        for sgn in (1, -1):
            kids.append([y * 1.55 + drift[0] + sgn * 1.3 * np.sin(th),
                         x * 1.55 + drift[1] + sgn * 1.3 * np.cos(th)])
            par.append(i + 1)
    B = np.c_[np.arange(1, 2 * n + 1), np.array(kids)]
    par = np.array(par)

    def draw(ax, A, Bshow, par, pred, title):
        ax.scatter(A[:, 2], A[:, 1], s=120, facecolor="none", edgecolor=INK, lw=1.2, zorder=3)
        ok = pred == par
        for (_, y, x), p, good in zip(Bshow, pred, ok):
            ay, ax_ = A[p - 1, 1:]
            ax.annotate("", (ax_, ay), (x, y), arrowprops=dict(arrowstyle="-|>", lw=1,
                        color=SERIES[2] if good else RED, shrinkA=2, shrinkB=6))
        ax.scatter(Bshow[:, 2], Bshow[:, 1], s=18, color=[SERIES[2] if g else RED for g in ok],
                   zorder=4)
        ax.set_title(f"{title}\n{ok.sum()}/{len(ok)} correct", fontsize=9, color=INK)
        ax.set_aspect("equal")
        _noaxis(ax)

    def near(src, dst):
        return src[cKDTree(src[:, 1:]).query(dst[:, 1:])[1], 0].astype(int)

    Bc = expansion_compensate(A, B)
    # capacity toy: parent 1 divided, and a neighbour's child drifted towards it
    A2 = np.array([[1, 0.0, 0.0], [2, 0.0, 6.0], [3, 5.0, 3.0]])
    B2 = np.array([[1, -0.3, -1.4], [2, 0.3, 1.4], [3, 0.2, 2.6], [4, 0.2, 8.0],
                   [5, 5.0, 2.2], [6, 5.2, 4.0]])
    par2 = np.array([1, 1, 2, 2, 3, 3])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 0.8, 0.8]})
    draw(axes[0], A, B, par, near(A, B), "(a) nearest: b → nearest cell at a\n(colony grew 1.55×)")
    draw(axes[1], A, Bc, par, near(A, Bc), "(b) nearest_rc: rescale b to a's centroid\nand RMS "
         "radius first")
    draw(axes[2], A2, B2, par2, near(A2, B2), "(c) nearest, crowded division:\nparent 1 gets 3 "
         "children")
    draw(axes[3], A2, B2, par2, _lap(A2, B2), "(d) LAP: each parent takes ≤ 2,\nsecond child "
         "costs extra")
    fig.text(0.01, 0.0, "Synthetic. Open circles = cells at a; dots = cells at b; arrow = "
             "predicted ancestor; red = wrong. (a–b): 14 cells each divide once, the colony "
             "expands and every family drifts randomly. (c–d): 3 parents, 6 children.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, path)


# ---------- result figures ----------

def fig_depth_acc(out, path):
    dd = pd.read_csv(out / "tables/expA_depth.csv")
    shown = ["s1", "s8", "s32", "ends"]
    bls = ["nearest_anchor", "nearest_anchor_rc", "chained_nearest", "chained_lap", "chained_lap_rc"]
    fig, axes = plt.subplots(1, len(shown), figsize=(14, 3.4), sharey=True)
    for ax, sidk in zip(axes, shown):
        _style(ax)
        w = 0.8 / len(bls)
        for i, bl in enumerate(bls):
            d = dd[(dd.baseline == bl) & (dd.schedule_id == sidk)]
            col, ls = BASELINE_STYLE[bl]
            ax.bar(d.depth + (i - len(bls) / 2 + 0.5) * w, d.acc, w, color=col,
                   hatch="///" if ls == "--" else None, edgecolor="white", lw=0.3,
                   label=bl.replace("_", " "))
        ax.set_xticks(range(4), ["0", "1", "2", "3+"])
        ax.set_title(f"schedule {sidk}", fontsize=9, color=INK)
        ax.set_xlabel("generations to ancestor", fontsize=8, color=MUTED)
    axes[0].set_ylabel("accuracy (OIAM, H = 132 min)", fontsize=8, color=MUTED)
    axes[0].set_ylim(0, 1)
    axes[-1].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def fig_density(out, cfgs, path):
    R = pd.read_parquet(out / "results/baselines_v0.parquet")
    H = max(cfgs["one_in_a_million"]["horizons_frames"])
    r = R[(R.dataset == "one_in_a_million") & (R.horizon_frames == H)
          & R.experiments.str.contains(f"A_h{H}") & (R.n_scored > 0)]
    r = r.assign(acc=r.n_correct / r.n_scored)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, sidk in zip(axes, ["s1", "ends"]):
        _style(ax)
        for bl in ["nearest_anchor", "nearest_anchor_rc", "chained_nearest", "chained_lap_rc"]:
            d = r[(r.baseline == bl) & (r.schedule_id == sidk)].sort_values("n_anchors")
            if bl.startswith("chained") and sidk == "ends":
                continue
            col, ls = BASELINE_STYLE[bl]
            ax.scatter(d.n_anchors, d.acc, s=10, color=col, alpha=0.5)
            ax.plot(d.n_anchors, d.acc.rolling(9, center=True, min_periods=3).mean(), color=col,
                    ls=ls, lw=1.8, label=bl.replace("_", " "))
        ax.set_xscale("log")
        ax.set_xlabel("cells at anchor frame a (log)", fontsize=8, color=MUTED)
        ax.set_title(f"OIAM, H = {H} min, schedule {sidk}: one dot per window", fontsize=9,
                     color=INK, loc="left")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("window accuracy", fontsize=8, color=MUTED)
    axes[0].set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, path)


def fig_qualitative(out, data, path, alias="M4", a=406, b=538):
    name, sid = SEQ_DIR[alias]
    s = Seq(out, name, sid)
    ws = Windows(out / "benchmarks/v0", data)
    byid = {w["window_id"]: w for w in ws.windows if w["sequence_id"] == alias}
    _, tg, to_gt = _window(out, alias, a, b)
    anc, col = _lineage_colors(tg, to_gt, a, b)
    mb, gb = s.mask(b), _norm(s.img(b))
    pb = s.obs[s.obs.frame_index == b]
    cy, cx = pb.center_y.median(), pb.center_x.median()
    r = int(max(pb.center_y.max() - pb.center_y.min(), pb.center_x.max() - pb.center_x.min()) / 2) + 40
    panels = [("truth", None), ("nearest_anchor", "ends"), ("nearest_anchor_rc", "ends"),
              ("chained_nearest", "s32"), ("chained_nearest", "s1"), ("chained_lap_rc", "s1")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.6))
    for ax, (bl, sidk) in zip(axes.flat, panels):
        if bl == "truth":
            lab, wrong, title = anc, set(), "ground truth (colour = true ancestor at a)"
        else:
            win = ws.load(byid[f"{alias}_a{a:04d}_b{b:04d}_{sidk}"])
            pred = BASELINES[bl](win)
            lab = {to_gt[(b, t)]: to_gt[(a, p)] for t, p in pred.items()}
            lab = {t: lab[t] for t in anc}
            wrong = {t for t in anc if lab[t] != anc[t]}
            title = (f"{bl} · schedule {sidk} ({len(win['window']['observed_frames'])} frames)\n"
                     f"{len(anc) - len(wrong)}/{len(anc)} correct; red outline = wrong ancestor")
        rgb = _overlay(gb, mb, {t: col[p] for t, p in lab.items()}, alpha=0.6, outline=wrong,
                       outline_color=(0.85, 0, 0))
        c, _ = _crop(rgb, cy, cx, r)
        ax.imshow(c)
        _noaxis(ax)
        ax.set_title(title, fontsize=8, color=INK)
    fig.suptitle(f"{alias}, a = {a} → b = {b} (132 min ≈ 2 cycles): predicted ancestor per cell "
                 f"at b", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_expB_all(out, path):
    eb = pd.read_csv(out / "tables/expB_all.csv")
    bls = ["nearest_anchor", "nearest_anchor_rc", "chained_lap", "chained_lap_rc"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.3), sharey=True)
    for ax, name in zip(axes, ORDER):
        _style(ax)
        for bl in bls:
            d = eb[(eb.dataset == name) & (eb.baseline == bl)]
            d = d[(d.n_scored >= 20) & (d.n_seq == d.n_seq.max())].sort_values("cycles")
            col, ls = BASELINE_STYLE[bl]
            ax.plot(d.cycles, d.acc, color=col, ls=ls, lw=1.6, marker="o", ms=3,
                    label=bl.replace("_", " ") + (" (= a→b LAP)" if "lap" in bl else ""))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=4))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if v >= 1 else f"1/{1 / v:g}"))
        ax.set_title(NAMES[name], fontsize=8, color=INK)
        ax.set_xlabel("gap (cell cycles)", fontsize=7, color=MUTED)
    axes[0].set_ylabel("accuracy, endpoints only", fontsize=8, color=MUTED)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    _save(fig, path)


def _displacement_rows(out, data, alias):
    """Per target of every Experiment-B window: true displacement / NN spacing, correctness."""
    root = out / "benchmarks/v0"
    ws = Windows(root, data)
    ws.windows = [w for w in ws.windows if w["sequence_id"] == alias and "B" in w["experiments"]]
    tg = pd.read_parquet(root / "labels/targets" / f"{alias}.parquet")
    tg = tg[(tg.target_status == "valid_anchor") & tg.in_foi]
    by = dict(list(tg.groupby("window_id")))
    rows = []
    for w in ws.windows:
        if w["window_id"] not in by:
            continue
        win = ws.load(w)
        a, b = w["anchor_frame"], w["target_frame"]
        da, db = win["detections"][a], win["detections"][b]
        if len(da) < 2:
            continue
        spacing = np.median(cKDTree(da[:, 1:]).query(da[:, 1:], k=2)[0][:, 1])
        pa = {int(i): (y, x) for i, y, x in da}
        pbm = {int(i): (y, x) for i, y, x in db}
        pn, prc = BASELINES["nearest_anchor"](win), BASELINES["nearest_anchor_rc"](win)
        t = by[w["window_id"]]
        for u, p in zip(t.target_detection_id.astype(int), t.ancestor_detection_id.astype(int)):
            rows.append({"alias": alias, "disp": np.hypot(*np.subtract(pbm[u], pa[p])) / spacing,
                         "nearest": pn[u] == p, "rc": prc[u] == p})
    return pd.DataFrame(rows)


def fig_displacement(out, data, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _style(ax)
    bins = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 1e9])
    for i, alias in enumerate(["M4", "SIM-02", "HSC-02", "MuSC-02", "HeLa-02"]):
        d = _displacement_rows(out, data, alias)
        d["bin"] = pd.cut(d.disp, bins, right=False)
        g = d.groupby("bin", observed=True).agg(acc=("nearest", "mean"), n=("nearest", "size"))
        g = g[g.n >= 30]
        mids = [iv.left for iv in g.index]
        ax.plot(np.arange(len(bins) - 1)[[list(pd.cut([m], bins, right=False).codes)[0]
                                          for m in mids]], g.acc, marker="o", ms=3, lw=1.6,
                color=SERIES[i], label=alias)
    ax.set_xticks(range(len(bins) - 1), ["<.25", ".25", ".5", ".75", "1", "1.5", "2", "3", "5",
                                         "≥10"], fontsize=7)
    ax.set_xlabel("true displacement ancestor(a) → target(b), in units of median "
                  "nearest-neighbour spacing at a", fontsize=8, color=MUTED)
    ax.set_ylabel("nearest_anchor accuracy", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Endpoints-only linking works while cells move less than about half the "
                 "spacing between cells", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def fig_siblings(out, data, path, alias="M4", b=538):
    root = out / "benchmarks/v0"
    name, sid = SEQ_DIR[alias]
    s = Seq(out, name, sid)
    ws = Windows(root, data)
    w = next(w for w in ws.windows if w["sequence_id"] == alias and w["target_frame"] == b
             and w["schedule_id"] == "ends")
    win = ws.load(w)
    sb = pd.read_parquet(root / "labels/siblings" / f"{alias}.parquet")
    sb = sb[sb.window_id == w["window_id"]]
    gt = {tuple(sorted(x)) for x in zip(sb.det_1, sb.det_2)}
    pred = mutual_nearest_siblings(win)
    P = {int(i): (y, x) for i, y, x in win["detections"][b]}
    pb = s.obs[s.obs.frame_index == b]
    cy, cx = pb.center_y.median(), pb.center_x.median()
    r = 150
    img, (y0, x0) = _crop(_norm(s.img(b)), cy, cx, r)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ax.imshow(img, cmap="gray")
    for pairs, col, lw, lab in [(gt, LIME, 3.5, "true direct sisters"),
                                (pred, SERIES[1], 1.3, "mutual nearest neighbours")]:
        first = True
        for i, j in pairs:
            (ya, xa), (yb, xb) = P[i], P[j]
            if all(0 <= v < 2 * r for v in [ya - y0, yb - y0, xa - x0, xb - x0]):
                ax.plot([xa - x0, xb - x0], [ya - y0, yb - y0], color=col, lw=lw,
                        label=lab if first else None, solid_capstyle="round")
                first = False
    _noaxis(ax)
    ax.legend(frameon=True, fontsize=7, loc="lower left")
    tp = len(gt & pred)
    ax.set_title(f"{alias} t = {b}: {len(gt)} true sister pairs, {len(pred)} MNN pairs, {tp} "
                 f"shared (whole frame)", fontsize=8, color=INK)
    ax = axes[1]
    _style(ax)
    t = pd.read_csv(out / "tables/siblings.csv")
    t = t.assign(k=t.dataset.map(ORDER.index)).sort_values(["k", "split"])
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.precision, 0.4, color=SERIES[0], label="precision")
    ax.bar(x + 0.2, t.recall, 0.4, color=SERIES[1], label="recall")
    short = {"one_in_a_million": "OIAM", "sim_plus": "SIM+", "hsc": "HSC", "musc": "MuSC",
             "hela": "HeLa"}
    ax.set_xticks(x, [f"{short[d]}\n{s_.replace('domain_', '')}" for d, s_ in
                      zip(t.dataset, t.split)], fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Sister-pair baseline (MNN at b), H = 1 cycle", fontsize=9, color=INK,
                 loc="left")
    fig.tight_layout()
    _save(fig, path)


def fig_domain(out, path):
    d = pd.read_csv(out / "tables/domain.csv")
    d = d[d.split.isin(["test", "domain_test", "sanity"])]
    seqs = ["M4", "SIM-01", "SIM-02", "HSC-02", "MuSC-02", "HeLa-02"]
    bls = [("nearest_anchor", "ends"), ("nearest_anchor_rc", "ends"), ("chained_nearest", "s1"),
           ("chained_lap", "s1"), ("chained_lap_rc", "s1")]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    _style(ax)
    w = 0.8 / len(bls)
    for i, (bl, sidk) in enumerate(bls):
        v = [d[(d.sequence_id == q) & (d.baseline == bl) & (d.schedule_id == sidk)].acc.mean()
             for q in seqs]
        col, ls = BASELINE_STYLE[bl]
        ax.bar(np.arange(len(seqs)) + (i - len(bls) / 2 + 0.5) * w, v, w, color=col,
               hatch="///" if ls == "--" else None, edgecolor="white", lw=0.3,
               label=f"{bl.replace('_', ' ')} · {sidk}")
    ax.set_xticks(range(len(seqs)), seqs, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy at H = 1 cycle", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    _save(fig, path)


def make_all(out: Path, data: Path, datasets: dict, only=None):
    """datasets: main.DATASETS (pixel size, time step); merged with benchmark config."""
    fd = out / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    cfgs = json.loads((out / "benchmarks/v0/config.json").read_text())
    for k, v in datasets.items():
        cfgs[k] = {**v, **cfgs.get(k, {})}
    jobs = {
        "fig01_gallery": lambda p: fig_gallery(out, cfgs, p),
        "fig02_counts": lambda p: fig_counts(out, cfgs, p),
        "fig03_cycles": lambda p: fig_cycles(out, cfgs, p),
        "fig04_quality": lambda p: fig_quality(out, p),
        "fig05_task": lambda p: fig_task(out, p),
        "fig06_lineage": lambda p: fig_lineage(out, p),
        "fig07_schedules": lambda p: fig_schedules(out, cfgs, p),
        "fig08_strata": lambda p: fig_strata(out, cfgs, p),
        "fig09_baselines": lambda p: fig_baseline_schematic(p),
        "fig10_depth": lambda p: fig_depth_acc(out, p),
        "fig11_density": lambda p: fig_density(out, cfgs, p),
        "fig12_qualitative": lambda p: fig_qualitative(out, data, p),
        "fig13_expB_all": lambda p: fig_expB_all(out, p),
        "fig14_displacement": lambda p: fig_displacement(out, data, p),
        "fig15_siblings": lambda p: fig_siblings(out, data, p),
        "fig16_domain": lambda p: fig_domain(out, p),
    }
    for k, f in jobs.items():
        if only is None or k in only:
            f(fd / f"{k}.png")
