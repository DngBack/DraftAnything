"""Figures from results/ (run from mllm_readout/: python paper/make_figures.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_tables import MODEL, ORDER, R

TASK = {"count": "Count (syn.)", "count_real": "CountBench", "spatial": "Spatial (syn.)", "spatial_real": "What'sUp",
        "attr": "Attribute (syn.)", "color_cf": "CounterFact"}
# categorical slots 1-3 of the default validated palette, fixed order
COL = {"bias": "#2a78d6", "loss": "#eb6834", "perc": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"


def decomposition():
    d = pd.concat([pd.read_csv(f"{R}/g15_table.csv"), pd.read_csv(f"{R}/g2_table.csv")]).drop_duplicates("cell")
    d["m"] = d.cell.map(lambda c: next(m for m in sorted(MODEL, key=len, reverse=True) if c.startswith(m + "_")))
    d["task"] = [c[len(m) + 1:] for c, m in zip(d.cell, d.m)]
    d = d[d.n_err >= 30]
    d = d.assign(t=d.task.map(list(TASK).index), o=d.m.map(ORDER.index)).sort_values(["t", "o"])
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(3.3, 3.9))
    y, ticks, labels, seps, last = 0, [], [], [], None
    for r in d.itertuples():
        if last is not None and r.task != last:
            y += 0.6
            seps.append(y - 0.8)
        left = 0
        for k in ("bias", "loss", "perc"):
            v = getattr(r, k)
            ax.barh(y, v, left=left, height=0.78, color=COL[k], edgecolor="white", linewidth=1)
            left += v
        ax.text(1.01, y, f"{r.n_err}", va="center", ha="left", fontsize=6, color=MUTED)
        ticks.append(y)
        labels.append(f"{TASK[r.task]} · {MODEL[r.m].replace('Qwen3-VL', 'Qwen3VL').replace('InternVL3.5', 'InternVL')}"
                      if r.task != last else MODEL[r.m].replace("Qwen3-VL", "Qwen3VL").replace("InternVL3.5", "InternVL"))
        last = r.task
        y += 1
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(1.01, -1.1, "#err", fontsize=6, color=MUTED, ha="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL]
    ax.legend(handles, ["bias (calibration fixes)", "loss (only probe fixes)", "perc (neither)"], loc="lower center",
              bbox_to_anchor=(0.35, 1.0), ncol=3, frameon=False, fontsize=6, handlelength=1, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig("paper/fig/decomp.pdf", bbox_inches="tight")
    fig.savefig("paper/fig/decomp.png", dpi=200, bbox_inches="tight")


def teaser():
    """Fig. 1: per cell, probe-model gap (x) vs. what recalibrating the model's own answer logits recovers (y)."""
    d = pd.concat([pd.read_csv(f"{R}/g15_table.csv"), pd.read_csv(f"{R}/g2_table.csv")]).drop_duplicates("cell")
    d["m"] = d.cell.map(lambda c: next(m for m in sorted(MODEL, key=len, reverse=True) if c.startswith(m + "_")))
    d["task"] = [c[len(m) + 1:] for c, m in zip(d.cell, d.m)]
    d = d[d.n_err >= 30]
    gx, gy = 100 * (d.probe - d.model), 100 * (d.calib - d.model)
    # task identity: fixed categorical order of the validated default palette (slots 1-6)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5fd3", "#d4a20f", "#e0457b"]
    mark = {"count": "o", "count_real": "s", "spatial": "^", "spatial_real": "D", "attr": "v", "color_cf": "P"}
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    lim = max(gx.max(), gy.max()) * 1.08
    xs = np.linspace(0, lim, 2)
    ax.fill_between(xs, xs / 2, xs, color="#2a78d6", alpha=0.07, lw=0)
    ax.plot(xs, xs, color=MUTED, lw=0.8, ls="--")
    ax.plot(xs, xs / 2, color=MUTED, lw=0.8, ls=":")
    ax.text(lim * 0.97, lim * 0.93, "calibration closes\nthe whole gap", ha="right", va="top", fontsize=6, color=MUTED)
    ax.text(lim * 0.62, lim * 0.29, "half the gap", ha="left", va="top", fontsize=6, color=MUTED, rotation=0)
    for k, t in enumerate(TASK):
        x = d.task == t
        ax.scatter(gx[x], gy[x], s=22, marker=mark[t], color=pal[k], edgecolor="white", linewidth=0.6, label=TASK[t], zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(min(-2, gy.min() - 1), lim)
    ax.set_xlabel("probe $-$ model accuracy (points)", color=INK)
    ax.set_ylabel("calibrated $-$ model (points)", color=INK)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    n_ok = int((d.cal_recovers >= 0.5).sum())
    ax.set_title(f"Recalibrating the output recovers $\\geq$half the probe gap in {n_ok}/{len(d)} cells", fontsize=7, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=5.5, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
    fig.tight_layout()
    fig.savefig("paper/fig/teaser.pdf", bbox_inches="tight")
    fig.savefig("paper/fig/teaser.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    decomposition()
    teaser()
