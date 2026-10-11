"""Paper figures from data and result files.  python paper/make_figures.py"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, SEQUENCE, examples  # noqa: E402
from experiments.lineage_glue import INDEX  # noqa: E402

RES = ROOT / "outputs/results"
FIG = Path(__file__).parent / "fig"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})


def distinct_colours(pos, k=10, seed=0):
    """tab20 colours such that each ancestor differs from its k nearest neighbours (greedy colouring)."""
    from scipy.spatial import cKDTree
    pos = np.asarray(pos, float)
    nb = cKDTree(pos).query(pos, k=min(k + 1, len(pos)))[1][:, 1:]
    adj = [set() for _ in pos]
    for i, row in enumerate(nb):
        for j in row:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    pal = np.random.default_rng(seed).permutation(20)
    out = np.full(len(pos), -1)
    for i in sorted(range(len(pos)), key=lambda i: -len(adj[i])):
        used = {out[j] for j in adj[i]}
        free = [c for c in pal if c not in used]
        out[i] = free[0] if free else pal[i % 20]
    return plt.cm.tab20(out / 19)


def blind4_effects():
    """Per-colony paired effects in blind test 4 (28 colonies, three subsets, run once)."""
    d = pd.read_parquet(RES / "blind_test4.parquet")
    seq = d.groupby(["dataset", "sequence", "gap", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    base = ["nearest_anchor", "centroid_translation", "radial_expansion", "radial_sector", "ot_exact", "ot_uot"]
    groups = (("zero-shot RodSim\n− best baseline", seq.sim - seq[base].max(axis=1)),
              ("bootstrapped\n− zero-shot", seq.adaptm - seq.sim),
              ("Hungarian\n− zero-shot", seq.adapt - seq.sim))
    colours = {"trpL": "#08519c", "metA": "#4292c6", "pheA": "#9ecae1"}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    stats = []
    for k, (name, v) in enumerate(groups):
        x = k + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v.values, s=9, c=[colours[ds] for ds in v.index.get_level_values(0)], linewidths=0, zorder=3)
        boot = [rng.choice(v.values, len(v)).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.errorbar(k + 0.32, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", ms=3, color="k", lw=1, capsize=2)
        stats.append((name.replace("\n", " "), v.median(), v.mean(), lo, hi, int((v > 0).sum()), len(v)))
    ax.axhline(0, color="#888888", lw=0.8, zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
    ax.set_ylabel("accuracy difference", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    for ds, c in colours.items():
        ax.scatter([], [], s=9, c=c, label=ds)
    ax.legend(frameon=False, fontsize=6, loc="upper right", handletextpad=0.1, borderaxespad=0.1)
    fig.savefig(FIG / "blind4_effects.pdf", bbox_inches="tight")
    fig.savefig(FIG / "blind4_effects.png", dpi=200, bbox_inches="tight")
    for s in stats:
        print("%-55s median %.3f mean %.3f CI [%.3f, %.3f] positive %d/%d" % s)


def teaser():
    windows = Windows(BENCH, ROOT / "data").windows
    e = next(x for x in examples("M3", windows, compensate=False, horizon=132)
             if x["window"]["window_id"] == "M3_a0281_b0413_ends")
    fr = pd.read_parquet(INDEX / SEQUENCE["M3"] / "frames.parquet").set_index("frame_index")
    anc, tgt, tg = e["anchor"], e["target"], e["targets"]
    lab = dict(zip(tg.target_detection_id, tg.ancestor_detection_id))
    # highlight eight families spread over the colony (farthest-point sampling among ancestors with >= 2
    # descendants); all other cells grey, so displacement and interleaving of families are visible
    ids = list(anc.local_detection_id)
    pa = anc[["center_y", "center_x"]].to_numpy()
    ndesc = pd.Series(list(lab.values())).value_counts()
    ok = [k for k, i in enumerate(ids) if ndesc.get(i, 0) >= 2]
    pick = [ok[int(np.argmin(np.linalg.norm(pa[ok] - pa[ok].mean(0), axis=1)))]]
    while len(pick) < 8:
        dist = np.min(np.linalg.norm(pa[ok][:, None] - pa[pick][None], axis=-1), 1)
        pick.append(ok[int(np.argmax(dist))])
    grey = (0.75, 0.75, 0.75, 0.55)
    colour = {i: grey for i in ids}
    colour.update({ids[k]: plt.cm.tab10([0, 1, 2, 3, 4, 5, 6, 8][n]) for n, k in enumerate(pick)})  # skip grey
    pb = tgt[["center_y", "center_x"]].to_numpy()
    lo, hi = pb.min(0) - 25, pb.max(0) + 25
    fig = plt.figure(figsize=(7.0, 2.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.32, 1.2], wspace=0.06)
    for k, (t, df, title) in enumerate(((e["window"]["anchor_frame"], anc, "(a) snapshot $a$"),
                                        (e["window"]["target_frame"], tgt, "(b) $b=a+132$ min"))):
        ax = fig.add_subplot(gs[k])
        im = tifffile.imread(fr.image_path[t]).astype(float)
        ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5))
        c = [colour[i] for i in df.local_detection_id] if k == 0 else \
            [colour.get(lab.get(i), (1, 1, 1, 0.0)) for i in df.local_detection_id]
        hl = np.array([ci != grey and ci[3] > 0 for ci in c])
        size = 16 if k == 0 else 11
        ax.scatter(df.center_x[~hl], df.center_y[~hl], s=size * 0.5, c=[c[i] for i in np.where(~hl)[0]], linewidths=0)
        ax.scatter(df.center_x[hl], df.center_y[hl], s=size * 1.6, c=[c[i] for i in np.where(hl)[0]],
                   linewidths=0.4, edgecolors="white")
        ax.set_xlim(lo[1], hi[1])
        ax.set_ylim(hi[0], lo[0])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8)
    ax = fig.add_subplot(gs[3])
    tr = pd.read_parquet(RES / "trackastra_gap.parquet")
    v3 = pd.read_parquet(RES / "image_partition_decode_raft_v3+raft_v3_s1+raft_v3_s2.parquet")  # Tab. 2 bold + oracle rows
    hs = [33, 66, 132]
    lines = {"Nearest ancestor": tr[tr.method == "nearest"].groupby("H").accuracy.mean(),
             "Trackastra, off the shelf": tr[tr.method == "trackastra_argmax"].groupby("H").accuracy.mean(),
             # Tab. 2 entry: per-gap model at 33/66, multi-gap model at 132 (chosen on M2; docs/matched_trackastra_protocol.md)
             "Trackastra, fine-tuned": pd.concat([
                 pd.read_parquet(RES / "trackastra_ft.parquet").query("method == 'trackastra_ft_argmax' and H != 132"),
                 pd.read_parquet(RES / "trackastra_ft_multi.parquet").query("method == 'trackastra_ft_argmax' and H == 132")])
             .groupby("H").accuracy.mean(),
             "Unbalanced OT": pd.read_parquet(RES / "ot_baseline.parquet").query("method == 'ot_uot_radial'")
             .groupby("H").accuracy.mean(),
             "LineageRAFT + genealogy (ens.)": v3.groupby("H").mf_img2.mean(),
             "Oracle family partition": v3.groupby("H").oracle_partition.mean()}
    style = {"Nearest ancestor": ("#888888", "o", "-"), "Unbalanced OT": ("#7a5195", "^", "-"), "Trackastra, off the shelf": ("#d95f02", "s", "--"),
             "Trackastra, fine-tuned": ("#d95f02", "s", "-"),
             "LineageRAFT + genealogy (ens.)": ("#1b6fb5", "D", "-"), "Oracle family partition": ("#1b6fb5", "", ":")}
    for name, s in lines.items():
        c, m, ls = style[name]
        ax.plot(hs, [s[h] for h in hs], color=c, marker=m, ms=3.5, ls=ls, lw=1.4, label=name)
    ax.set_xticks(hs)
    ax.set_xlabel("gap between snapshots (min)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6, loc="lower center",
               bbox_to_anchor=(0.32, -0.09), ncol=2, handlelength=1.6, labelspacing=0.25, columnspacing=1.0)
    ax.set_title("(c) OIAM M3/M4", fontsize=8)
    fig.savefig(FIG / "teaser.pdf", bbox_inches="tight")
    fig.savefig(FIG / "teaser.png", dpi=200, bbox_inches="tight")


def density():
    """Accuracy against the number of ancestors in the window, per gap."""
    tr = pd.read_parquet(RES / "trackastra_gap.parquet")
    v3 = pd.read_parquet(RES / "image_partition_decode_raft_v3.parquet")
    x = tr[tr.method == "nearest"][["window_id", "H", "n_anchors", "accuracy"]].rename(columns={"accuracy": "near"})
    x = x.merge(tr[tr.method == "trackastra_argmax"][["window_id", "accuracy"]].rename(columns={"accuracy": "ta"}),
                on="window_id").merge(
        pd.read_parquet(RES / "trackastra_ft.parquet").query("method == 'trackastra_ft_argmax'")[["window_id", "accuracy"]]
        .rename(columns={"accuracy": "ft"}), on="window_id").merge(v3[["window_id", "mf_img2", "argmax", "oracle_partition"]], on="window_id")
    edges = np.array([0, 25, 50, 100, 200, 400, 2000])
    x["bin"] = np.digitize(x.n_anchors, edges[1:-1])
    mids = np.sqrt(np.maximum(edges[:-1], 12) * edges[1:])
    series = (("near", "Nearest ancestor", "#888888", "o", "-"), ("ta", "Trackastra, off the shelf", "#d95f02", "s", "--"),
              ("ft", "Trackastra, fine-tuned", "#d95f02", "s", "-"),
              ("argmax", "LineageRAFT", "#7fb2e5", "^", "-"), ("mf_img2", "+ genealogy decoder", "#1b6fb5", "D", "-"),
              ("oracle_partition", "Oracle partition", "#1b6fb5", "", ":"))
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.55), sharey=True)
    for ax, h in zip(axes, (33, 66, 132)):
        g = x[x.H == h].groupby("bin")
        for col, name, c, m, ls in series:
            v = g[col].mean()
            ax.plot(mids[v.index], v.values, color=c, marker=m, ms=2.5, ls=ls, lw=1.1, label=name)
        ax.set_xscale("log")
        ax.set_xticks([20, 100, 500])
        ax.set_xticklabels(["20", "100", "500"])
        ax.set_title(f"{h} min", fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("accuracy", fontsize=7)
    axes[1].set_xlabel("ancestors in window", fontsize=7, labelpad=1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=5.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08),
               handlelength=1.6, columnspacing=1.0)
    fig.savefig(FIG / "density.pdf", bbox_inches="tight")
    fig.savefig(FIG / "density.png", dpi=200, bbox_inches="tight")


def qualitative(window="M4_a0604_b0670_ends"):
    """Ground-truth families against predicted families on one dense window; errors marked."""
    f = RES / f"qualitative_{window}.npz"
    if not f.exists():
        return
    d = np.load(f)
    pa, pb, sc, truth = d["anchor_pos"], d["target_pos"], d["scored"], d["truth"]
    rng = np.random.default_rng(1)
    col = distinct_colours(pa, seed=1)
    half = 0.13 * (pb[sc].max(0) - pb[sc].min(0)).max()
    # crop where the genealogy decoder fixes most errors of the network
    fixed = (d["ours"][sc] != truth) & (d["ours_potts"][sc] == truth)
    q = pb[sc]
    inside = (np.abs(q[:, None] - q[None]) <= half * 0.92).all(-1)
    c = q[int(np.argmax((inside & fixed[None]).sum(1)))]
    lo, hi = c - half, c + half
    im = d["target_image"]
    panels = [("Ground truth", truth), ("Nearest", d["nearest"][sc]),
              ("LineageRAFT", d["ours"][sc]), ("+ genealogy", d["ours_potts"][sc])]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
    for ax, (title, pred) in zip(axes, panels):
        ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5))
        wrong = pred != truth
        ax.scatter(pb[sc, 1], pb[sc, 0], s=30, c=col[pred], linewidths=0)
        if title != "Ground truth":
            ax.scatter(pb[sc][wrong, 1], pb[sc][wrong, 0], s=46, facecolors="none", edgecolors="red", linewidths=0.9)
            if title.startswith("+"):
                ax.scatter(pb[sc][fixed, 1], pb[sc][fixed, 0], s=46, facecolors="none", edgecolors="#2ca02c",
                           linewidths=1.1)
            title = f"{title} ({100 * (1 - wrong.mean()):.0f}%)"
        ax.set_xlim(lo[1], hi[1])
        ax.set_ylim(hi[0], lo[0])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=7)
    fig.subplots_adjust(wspace=0.03)
    fig.savefig(FIG / "qualitative.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG / "qualitative.png", dpi=200, bbox_inches="tight")


def robustness():
    """Accuracy under simulated detection errors (one panel per perturbation kind)."""
    f = RES / "detection_robustness.parquet"
    if not f.exists():
        return
    df = pd.read_parquet(f)
    df = df[df.H == 33]
    kinds = (("jitter", "position noise (spacings)"), ("drop", "missed detections"),
             ("merge", "merged cells"), ("spurious", "false detections"))
    series = (("nearest", "Nearest ancestor", "#888888", "o"), ("nearest_comp", "Radial expansion", "#555555", "s"),
              ("raft_sim_loc_argmax", "\\ours, sim only", "#e08b2c", "^"),
              ("raft_v3_argmax", "\\ours, OIAM-trained", "#1b6fb5", "D"))
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.6), sharey=True)
    clean = df[df.kind == "clean"].groupby("method").accuracy.mean()
    for ax, (k, label) in zip(axes, kinds):
        x = df[df.kind == k].groupby(["method", "level"]).accuracy.mean()
        for m, name, c, mk in series:
            if m not in clean:
                continue
            lv = [0.0] + list(x[m].index)
            ax.plot(lv, [clean[m]] + list(x[m].values), color=c, marker=mk, ms=3, lw=1.2,
                    label=name.replace("\\ours", "LineageRAFT"))
        ax.set_xlabel(label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(FIG / "robustness.pdf", bbox_inches="tight")
    fig.savefig(FIG / "robustness.png", dpi=200, bbox_inches="tight")


def sparsity():
    """Ancestry accuracy over H minutes against the number of snapshots taken in between."""
    d = pd.concat([pd.read_parquet(RES / f) for f in ("sparsity_curve.parquet", "sparsity_curve_nolabel.parquet",
                                                            "sparsity_curve_trackastra.parquet")
                   if (RES / f).exists()])
    series = (("hungarian", "Frame-to-frame linking", "#888888", "o", "-"),
              ("trackastra_argmax", "Trackastra (true masks)", "#d95f02", "D", "-"),
              ("raft_sim_loc", "LineageRAFT, simulation", "#9ecae1", "v", "-"),
              ("raft_pseudo", "LineageRAFT, self-labelled", "#4292c6", "s", "-"),
              ("lineageraft", "LineageRAFT, annotated", "#08519c", "^", "--"))
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.45), sharey=True)
    for ax, H in zip(axes, (66, 132)):
        g = d[d.H == H].assign(c=lambda x: x.accuracy * x.n).groupby(["method", "snapshots"])[["c", "n"]].sum()
        acc = (g.c / g.n)
        for key, name, c, m, ls in series:
            if key in acc.index.get_level_values(0):
                v = acc.loc[key]
                ax.plot(v.index, v.values, color=c, marker=m, ms=2.5, ls=ls, lw=1.1, label=name)
        ax.set_xscale("log")
        ticks = [2, 4, 12, 34, 67] if H == 66 else [2, 5, 13, 34, 133]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"ancestry over {H} min", fontsize=7.5)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("snapshots", fontsize=7.5, labelpad=1)
    axes[0].set_ylabel("accuracy", fontsize=7.5)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.06),
               handlelength=1.6, columnspacing=1.0)
    fig.savefig(FIG / "sparsity.pdf", bbox_inches="tight")
    fig.savefig(FIG / "sparsity.png", dpi=200, bbox_inches="tight")


def centre_trajectory(window="M4_a0604_b0670_ends", tag="raft_v3"):
    """Family centres through the refinement rounds: one dense window, and centre error / candidate recall per round."""
    f, g = RES / f"centre_trajectory_{window}.npz", RES / f"centre_trajectory_{tag}.parquet"
    if not (f.exists() and g.exists()):
        return
    d = np.load(f)
    cs, pb, sc, truth = d["centres"], d["target_pos"], d["scored"], d["truth"]
    ids, cen, s = d["family_ids"], d["family_centroid"], float(d["spacing"])
    fig = plt.figure(figsize=(7.0, 2.3))
    ax = fig.add_axes([0.0, 0.0, 0.36, 1.0])
    q = RES / f"qualitative_{window}.npz"
    if q.exists():
        im = np.load(q)["target_image"]
        ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5))
    # crop: the 14 s x 14 s box with the most families whose initial centre is off by > 1.5 spacings
    err0 = np.linalg.norm(cs[0][ids] - cen, axis=1) / s
    half = 7 * s
    inside = (np.abs(cen[:, None] - cen[None]) <= half * 0.8).all(-1)
    c = cen[int(np.argmax((inside & (err0 > 1.5)[None]).sum(1)))]
    lo, hi = c - half, c + half
    # the 8 families with initial error > 1.5 spacings closest to the crop centre (not chosen by outcome)
    near = np.argsort(np.linalg.norm(cen - c, axis=1))
    sel = [k for k in near if err0[k] > 1.5][:8]
    col = plt.cm.tab10([0, 1, 2, 3, 4, 5, 6, 8])
    ax.scatter(pb[:, 1], pb[:, 0], s=6, c=[(0.8, 0.8, 0.8, 0.8)], linewidths=0)
    for n, k in enumerate(sel):
        a = ids[k]
        m = sc[truth == a]
        ax.scatter(pb[m, 1], pb[m, 0], s=14, color=col[n], linewidths=0)
        path = cs[:, a]
        ax.plot(path[:, 1], path[:, 0], "-", color="white", lw=2.2, zorder=3)
        ax.plot(path[:, 1], path[:, 0], "-", color=col[n], lw=1.1, zorder=4)
        ax.scatter(path[0, 1], path[0, 0], s=16, marker="o", facecolors="white", edgecolors=col[n], lw=1, zorder=5)
        ax.scatter(path[-1, 1], path[-1, 0], s=22, marker="o", color=col[n], edgecolors="white", lw=0.6, zorder=5)
        ax.scatter(cen[k, 1], cen[k, 0], s=46, marker="X", color=col[n], edgecolors="black", lw=0.7, zorder=6)
    ax.set_xlim(lo[1], hi[1])
    ax.set_ylim(hi[0], lo[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) centres $c^{(0)}$ (open) $\\to c^{(4)}$ (filled); true centroid $\\mathbf{\\times}$", fontsize=7)
    df = pd.read_parquet(g)
    df["dense"] = df.n_anchors > 200
    cols = {33: "#1b9e77", 66: "#d95f02", 132: "#7570b3"}
    for pos, val, lab in (([0.45, 0.17, 0.24, 0.7], "centre_err_mean", "(b) centre error (spacings)"),
                          ([0.75, 0.17, 0.24, 0.7], "recall_k", "(c) true ancestor among 24 nearest")):
        a2 = fig.add_axes(pos)
        for h, cl in cols.items():
            for dense, ls in ((False, "-"), (True, "--")):
                x = df[(df.H == h) & (df.dense if dense else True)].groupby("round")[val].mean()
                a2.plot(x.index, x.values, ls, color=cl, marker="o", ms=2.5, lw=1,
                        label=f"{h} min" if not dense else None)
        a2.set_xticks(range(5))
        a2.set_xlabel("refinement round $t$", fontsize=7)
        a2.set_title(lab, fontsize=7)
        a2.tick_params(labelsize=6)
        a2.grid(alpha=0.3, lw=0.5)
        if val == "centre_err_mean":
            a2.legend(fontsize=6, frameon=False, title="solid: all, dashed: dense", title_fontsize=6)
    fig.savefig(FIG / "centre_trajectory.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG / "centre_trajectory.png", dpi=200, bbox_inches="tight")


def centre_inset(window="M4_a0604_b0670_ends", n_fam=4, half_spacings=5.0):
    """Small inset for the method figure: centres c^(0) -> c^(T) on real data, true family centroids."""
    f, q = RES / f"centre_trajectory_{window}.npz", RES / f"qualitative_{window}.npz"
    if not (f.exists() and q.exists()):
        return
    d = np.load(f)
    cs, pb, sc, truth = d["centres"], d["target_pos"], d["scored"], d["truth"]
    ids, cen, s = d["family_ids"], d["family_centroid"], float(d["spacing"])
    err0 = np.linalg.norm(cs[0][ids] - cen, axis=1) / s
    half = half_spacings * s
    inside = (np.abs(cen[:, None] - cen[None]) <= half * 0.75).all(-1)
    c = cen[int(np.argmax((inside & (err0 > 1.5)[None]).sum(1)))]
    near = np.argsort(np.linalg.norm(cen - c, axis=1))
    sel = [k for k in near if err0[k] > 1.5][:n_fam]  # not chosen by outcome
    col = plt.cm.tab10([0, 1, 2, 3, 4, 6, 8, 9])
    im = np.load(q)["target_image"]
    fig, ax = plt.subplots(figsize=(1.45, 1.45))
    ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5))
    for n, k in enumerate(sel):
        a = ids[k]
        m = sc[truth == a]
        ax.scatter(pb[m, 1], pb[m, 0], s=9, color=col[n], linewidths=0)
        path = cs[:, a]
        ax.plot(path[:, 1], path[:, 0], "-", color="white", lw=1.8, zorder=3)
        ax.plot(path[:, 1], path[:, 0], "-", color=col[n], lw=0.9, zorder=4)
        ax.scatter(path[0, 1], path[0, 0], s=10, facecolors="white", edgecolors=col[n], lw=0.8, zorder=5)
        ax.scatter(path[-1, 1], path[-1, 0], s=14, color=col[n], edgecolors="white", lw=0.5, zorder=5)
        ax.scatter(cen[k, 1], cen[k, 0], s=26, marker="X", color=col[n], edgecolors="black", lw=0.5, zorder=6)
    pts = np.concatenate([np.concatenate([cs[:, ids[k]], cen[k][None], pb[sc[truth == ids[k]]]]) for k in sel])
    mid, r = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2 + 0.8 * s
    ax.set_xlim(mid[1] - r, mid[1] + r)
    ax.set_ylim(mid[0] + r, mid[0] - r)
    ax.set_axis_off()
    fig.savefig(FIG / "centre_inset.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    fig.savefig(FIG / "centre_inset.png", bbox_inches="tight", pad_inches=0, dpi=200)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        FIG.mkdir(exist_ok=True)
        for name in sys.argv[1:]:
            globals()[name]()
        sys.exit()
    FIG.mkdir(exist_ok=True)
    teaser()
    density()
    qualitative()
    robustness()
