"""Static figures for the EDA report. Every function takes analysed data and returns a Figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from .data import FIELD_GROUPS, FIELD_TO_GROUP  # noqa: E402

# Reference palette (dataviz skill, light mode).
SHARE = "#257de0"      # categorical slot 1 - "share"
WITHHOLD = "#e3e3e3"   # neutral - "withhold"
TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e8e8e8"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9dc8f7", "#257de0", "#0e3d7e"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def _hgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def norm_heatmap(norm: pd.DataFrame) -> plt.Figure:
    """Scenario x field share/withhold grid, columns grouped by field group."""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(norm.values, cmap=ListedColormap([WITHHOLD, SHARE]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(norm.shape[1]), norm.columns, rotation=60, ha="right")
    ax.set_yticks(range(norm.shape[0]), norm.index)
    ax.set_xticks(np.arange(-0.5, norm.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, norm.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    # Group separators + group names on top.
    start = 0
    for group, fields in FIELD_GROUPS.items():
        n = sum(f in norm.columns for f in fields)
        if start:
            ax.axvline(start - 0.5, color=TEXT_2, linewidth=1.2)
        ax.text(start + n / 2 - 0.5, -0.8, group, ha="center", va="bottom", color=TEXT_2, fontsize=8)
        start += n
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Contextual norm: which field may be shared in which scenario", pad=22)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (SHARE, WITHHOLD)]
    ax.legend(handles, ["share (label=1)", "withhold (label=0)"], loc="upper left",
              bbox_to_anchor=(1.005, 1), frameon=False)
    return fig


def share_count_by_field(norm: pd.DataFrame) -> plt.Figure:
    """How many of the 8 scenarios allow each field."""
    counts = norm.sum(axis=0).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.barh(counts.index, counts.values, color=SHARE, height=0.6)
    for y, v in enumerate(counts.values):
        ax.text(v + 0.08, y, int(v), va="center", color=TEXT_2, fontsize=8)
    ax.set_yticks(range(len(counts)), [f"{f}  ·  {FIELD_TO_GROUP[f]}" for f in counts.index])
    ax.set_xlim(0, norm.shape[0])
    ax.set_xlabel(f"scenarios that allow sharing (of {norm.shape[0]})")
    ax.set_title("Shareability of each profile field")
    _hgrid(ax)
    return fig


def share_count_by_scenario(norm: pd.DataFrame, domains: pd.Series) -> plt.Figure:
    counts = norm.sum(axis=1).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(counts.index, counts.values, color=SHARE, height=0.6)
    for y, (s, v) in enumerate(counts.items()):
        ax.text(v + 0.15, y, f"{int(v)}  ({domains[s]})", va="center", color=TEXT_2, fontsize=8)
    ax.set_xlim(0, norm.shape[1])
    ax.set_xlabel(f"fields allowed (of {norm.shape[1]})")
    ax.set_title("Fields the agent may share, per scenario")
    _hgrid(ax)
    return fig


def similarity_heatmap(sim: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(sim.values.astype(float), cmap=SEQ_BLUE, vmin=0, vmax=1)
    ax.set_xticks(range(len(sim)), sim.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(sim)), sim.index)
    for i in range(len(sim)):
        for j in range(len(sim)):
            v = float(sim.iat[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.6 else TEXT)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Jaccard of shareable-field sets")
    ax.set_title("Overlap of shareable fields between scenarios")
    return fig


def leak_surface_bars(surface: pd.DataFrame) -> plt.Figure:
    s = surface["matchable_rate"].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.barh(s.index, s.values, color=SHARE, height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", color=TEXT_2, fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("withheld values detectable by verbatim string match")
    ax.set_title("Leak-detection surface per field")
    _hgrid(ax)
    return fig


def profile_demographics(profiles: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.hist(profiles["age"], bins=range(10, 90, 10), color=SHARE, rwidth=0.85)
    ax.set_xlabel("age")
    ax.set_ylabel("profiles")
    ax.set_title("Age")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for ax, col in zip(axes[1:], ["gender", "relationship_status"]):
        vc = profiles[col].value_counts().sort_values()
        ax.barh(vc.index, vc.values, color=SHARE, height=0.6)
        for y, v in enumerate(vc.values):
            ax.text(v + 0.1, y, v, va="center", color=TEXT_2, fontsize=8)
        ax.set_title(col.replace("_", " ").capitalize())
        _hgrid(ax)
    fig.suptitle(f"The {len(profiles)} synthetic user profiles", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig
