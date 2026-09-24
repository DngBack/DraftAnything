"""Aggregate audit, strata and baseline results into report tables + figures."""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# Reference categorical palette, first slots (dataviz skill, validated light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["one_in_a_million", "sim_plus", "hsc", "musc", "hela"]
# method family -> colour; *_rc (expansion-compensated) variants drawn dashed
BASELINE_STYLE = {
    "nearest_anchor": (SERIES[0], "-"), "nearest_anchor_rc": (SERIES[0], "--"),
    "chained_nearest": (SERIES[1], "-"), "chained_nearest_rc": (SERIES[1], "--"),
    "chained_lap": (SERIES[2], "-"), "chained_lap_rc": (SERIES[2], "--"),
}


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _sched_key(s):
    return (0, int(s[1:])) if s.startswith("s") else (1, 0) if s == "ends" else (2, int(s[4:]))


def load(out: Path):
    b = out / "benchmarks/v0"
    W = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(str(b / "inputs/windows/*.jsonl")))
                      for l in open(f)])
    T = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(b / "labels/targets/*.parquet"))))
    S = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(b / "labels/siblings/*.parquet"))))
    R = pd.read_parquet(out / "results/baselines_v0.parquet")
    cfg = json.loads((b / "config.json").read_text())
    R["acc"] = R.n_correct / R.n_scored.replace(0, np.nan)
    return W, T, S, R, cfg


def tables(root: Path):
    """Read root/{audit,benchmarks,results}; write root/tables/*.csv and root/figures/*.png."""
    out, fig_dir = root / "tables", root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    W, T, S, R, cfg = load(root)
    res = {}

    # 1. audit per movie
    aud = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(root / "audit/*/summary.csv"))))
    res["audit"] = aud
    # 2. horizons
    res["horizons"] = pd.DataFrame(cfg).T

    # 3. strata on unique (a, b) of experiment A (schedule-independent fields)
    wa = W[W.experiments.map(lambda e: any(x.startswith("A") for x in e)) & (W.schedule_id == "ends")]
    t = T.merge(wa[["window_id", "dataset", "split", "horizon_frames", "horizon_min"]], on="window_id")
    t["depth"] = t.generation_depth.clip(upper=3)
    g = t.groupby(["dataset", "split", "horizon_frames", "horizon_min"])
    st = pd.DataFrame({
        "windows": g.window_id.nunique(),
        "anchors_per_window": R[(R.schedule_id == "ends") & R.window_id.isin(wa.window_id)
                                & (R.baseline == "nearest_anchor")]
        .groupby(["dataset", "split", "horizon_frames", "horizon_min"]).n_anchors.mean().round(1),
        "targets": g.size(),
        "unresolved": g.target_status.apply(lambda s: int((s != "valid_anchor").sum())),
        "outside_foi": g.in_foi.apply(lambda s: int((~s).sum())),
    })
    ok = t[t.target_status == "valid_anchor"]
    st = st.join(ok.groupby(["dataset", "split", "horizon_frames", "horizon_min"]).depth
                 .value_counts(normalize=True).unstack(fill_value=0).mul(100).round(1)
                 .rename(columns=lambda d: f"depth{d}{'+' if d == 3 else ''}_%"))
    st = st.join(S.merge(wa[["window_id", "dataset", "split", "horizon_frames", "horizon_min"]],
                         on="window_id").groupby(["dataset", "split", "horizon_frames",
                                                  "horizon_min"]).size().rename("sibling_pairs"))
    res["strata"] = st.reset_index()

    # 4. hidden intermediate tracks by schedule (OIAM, longest horizon)
    Hmax = max(cfg["one_in_a_million"]["horizons_frames"])
    wh = W[(W.dataset == "one_in_a_million") & (W.horizon_frames == Hmax)
           & W.experiments.map(lambda e: f"A_h{Hmax}" in e)]
    th = T.merge(wh[["window_id", "schedule_id", "observed_frames"]], on="window_id")
    th = th[th.target_status == "valid_anchor"]
    hid = th.groupby("schedule_id").agg(
        observed_frames=("observed_frames", lambda s: len(s.iat[0])),
        pct_targets_with_hidden=("hidden_intermediates", lambda s: round(100 * (s > 0).mean(), 1)),
        mean_hidden=("hidden_intermediates", lambda s: round(s.astype(float).mean(), 2)))
    res["hidden"] = hid.loc[sorted(hid.index, key=_sched_key)].reset_index()

    # 5. experiment A accuracy, OIAM: macro over windows per movie, then over movies
    ra = R[(R.dataset == "one_in_a_million")
           & R.apply(lambda r: f"A_h{r.horizon_frames}" in r.experiments.split(","), axis=1)]
    per_movie = ra.groupby(["baseline", "horizon_frames", "schedule_id", "sequence_id", "split"]) \
        .acc.mean().reset_index()
    acc = per_movie.pivot_table(index=["baseline", "horizon_frames", "schedule_id"],
                                columns="sequence_id", values="acc")
    acc["macro_M0-M4"] = acc.mean(axis=1)
    acc["n_observed"] = ra.groupby(["baseline", "horizon_frames", "schedule_id"]).n_observed.first()
    acc = acc.reset_index()
    acc["_k"] = acc.schedule_id.map(_sched_key)
    res["expA_acc"] = acc.sort_values(["baseline", "horizon_frames", "_k"]).drop(columns="_k")

    # 5b. by generation depth (pooled within movie, macro over movies), longest horizon
    rd = ra[ra.horizon_frames == Hmax]
    rows = []
    for (bl, sid, seq), g in rd.groupby(["baseline", "schedule_id", "sequence_id"]):
        for d in range(4):
            n = g[f"n_scored_d{d}"].sum()
            if n:
                rows.append({"baseline": bl, "schedule_id": sid, "sequence_id": seq, "depth": d,
                             "acc": g[f"n_correct_d{d}"].sum() / n, "n": n})
    dd = pd.DataFrame(rows).groupby(["baseline", "schedule_id", "depth"]).agg(
        acc=("acc", "mean"), n=("n", "sum")).reset_index()
    dd["_k"] = dd.schedule_id.map(_sched_key)
    res["expA_depth"] = dd.sort_values(["baseline", "_k", "depth"]).drop(columns="_k")

    # 5c. descendant-count MAE per anchor (pooled within movie, macro over movies)
    cm = ra.groupby(["baseline", "horizon_frames", "schedule_id", "sequence_id"])[
        ["count_abs_err_sum", "count_n_anchors"]].sum()
    cm = (cm.count_abs_err_sum / cm.count_n_anchors).groupby(
        ["baseline", "horizon_frames", "schedule_id"]).mean().rename("count_MAE").reset_index()
    cm["_k"] = cm.schedule_id.map(_sched_key)
    res["expA_count"] = cm.sort_values(["baseline", "horizon_frames", "_k"]).drop(columns="_k")

    # 6. experiment B (endpoints only, growing delta), per dataset, macro over sequences.
    # On an "ends" window every chained baseline is a single a -> b link.
    rb = R[R.experiments.str.contains("B")].copy()
    rb["cycles"] = rb.apply(lambda r: r.horizon_frames / cfg[r.dataset]["cell_cycle_median_frames"],
                            axis=1)
    keys = ["baseline", "dataset", "horizon_frames", "horizon_min", "cycles"]
    ebb = rb.groupby(keys + ["sequence_id"]).agg(acc=("acc", "mean"), n=("n_scored", "sum")) \
        .reset_index().groupby(keys).agg(acc=("acc", "mean"), n_scored=("n", "sum"),
                                         n_seq=("sequence_id", "nunique")).reset_index()
    res["expB_all"] = ebb
    eb = ebb[ebb.baseline == "nearest_anchor"].drop(columns="baseline")
    res["expB"] = eb

    # 7. sibling baseline (mutual nearest neighbours at b), pooled per dataset/split at H = cycle
    rs = R[(R.baseline == "nearest_anchor") & (R.schedule_id == "ends")
           & R.apply(lambda r: r.horizon_frames == cfg[r.dataset]["cell_cycle_median_frames"]
                     and f"A_h{r.horizon_frames}" in r.experiments, axis=1)]
    sb = rs.groupby(["dataset", "split"])[["sib_gt", "sib_pred", "sib_tp"]].sum()
    sb["precision"] = (sb.sib_tp / sb.sib_pred).round(3)
    sb["recall"] = (sb.sib_tp / sb.sib_gt).round(3)
    res["siblings"] = sb.reset_index()

    # 8. cross-domain summary at H = one cell cycle: dense vs endpoints
    rx = R[R.apply(lambda r: r.horizon_frames == cfg[r.dataset]["cell_cycle_median_frames"]
                   and f"A_h{r.horizon_frames}" in r.experiments, axis=1)
           & R.schedule_id.isin(["s1", "ends"])]
    dom = rx.groupby(["dataset", "split", "baseline", "schedule_id", "sequence_id"]).agg(
        acc=("acc", "mean"), windows=("window_id", "nunique"), n=("n_scored", "sum")).reset_index()
    res["domain"] = dom

    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)

    _fig_expA(res["expA_acc"], cfg, fig_dir / "expA_oiam_accuracy.png")
    _fig_expB(eb, fig_dir / "expB_endpoints_accuracy.png")
    return res


def _fig_expA(acc, cfg, path):
    Hs = cfg["one_in_a_million"]["horizons_frames"]
    fig, axes = plt.subplots(1, len(Hs), figsize=(11, 3.4), sharey=True)
    for ax, H in zip(axes, Hs):
        _style(ax)
        for bl, (col, ls) in BASELINE_STYLE.items():
            d = acc[(acc.baseline == bl) & (acc.horizon_frames == H)
                    & (acc.schedule_id.str.startswith("s") | (acc.schedule_id == "ends"))]
            d = d.assign(stride=d.schedule_id.map(lambda s: H if s == "ends" else int(s[1:])))
            d = d.sort_values("stride")
            ax.plot(d.stride, d["macro_M0-M4"], color=col, ls=ls, lw=1.6, marker="o", ms=3,
                    label=bl.replace("_", " "))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(f"horizon {H} min ({H / cfg['one_in_a_million']['cell_cycle_median_frames']:.1f} cycles)",
                     fontsize=9, color=INK)
        ax.set_xlabel("stride between kept frames (min); rightmost = endpoints only", fontsize=7,
                      color=MUTED)
    axes[0].set_ylabel("ancestor accuracy (macro M0–M4)", fontsize=8, color=MUTED)
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("One-in-a-Million, experiment A: same (a, b), fewer kept frames", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_expB(eb, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _style(ax)
    for i, ds in enumerate(ORDER):
        d = eb[eb.dataset == ds]
        d = d[(d.n_scored >= 20) & (d.n_seq == d.n_seq.max())].sort_values("cycles")
        if len(d):
            ax.plot(d.cycles, d.acc, color=SERIES[i], lw=2, marker="o", ms=4, label=ds)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=4))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"1/{1 / v:g}"))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("gap between the two frames (in median cell cycles of that dataset)", fontsize=8,
                  color=MUTED)
    ax.set_ylabel("nearest-anchor accuracy", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("Experiment B: endpoints only, growing gap (≥20 scored targets, all movies)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
