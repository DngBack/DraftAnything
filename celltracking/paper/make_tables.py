"""Generate the paper's LaTeX tables from outputs/results/*.parquet.

Missing result files produce a \\TODO cell so the paper always compiles.
  python paper/make_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
RES = ROOT / "outputs/results"
OUT = Path(__file__).parent / "tables"
H = (33, 66, 132)


def load(name):
    p = RES / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def fmt(v, bold=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\TODO{}"
    s = f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"
    return rf"\textbf{{{s}}}" if bold else s


SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
SKIP_NOTE = ("Decoding skips the one 33-minute window with fewer than nine target cells, "
             "which moves 33-minute argmax values by up to 0.009 from other tables.")
PSEUDO_SEEDS = ("raft_v3p", "raft_v3p_s1", "raft_v3p_s2")
SEED_STD = {}
FRAMES = {}


def oiam_rows():
    """method -> (acc[H], dense acc[H]) on OIAM M3/M4 windows."""
    rows = {}

    def add(name, df, col="accuracy", hcol="H"):
        if df is None or not len(df):
            rows[name] = None
            return
        d = df.copy()
        d["dense"] = d.n_anchors > 200
        # per-window values of every row (one frame per seed), for the protocol audit (experiments/table2_audit.py)
        FRAMES.setdefault(name, []).append(
            d[[hcol, "window_id", "n_anchors", col] + (["n_scored"] if "n_scored" in d else [])]
            .rename(columns={hcol: "H", col: "value"}))
        rows[name] = ([d[d[hcol] == h][col].mean() for h in H],
                      [d[(d[hcol] == h) & d.dense][col].mean() for h in H])

    tr = load("trackastra_gap")
    if tr is not None:
        add("Nearest ancestor", tr[tr.method == "nearest"])
    ext = load("raft_external_darcy_auto")
    if ext is not None:
        o = ext[ext.dataset == "OIAM"].rename(columns={"gap_minutes": "H"})
        add("Radial expansion", o[o.method == "radial_expansion"])
        add("Darcy flow prior (ours)", o[o.method == "darcy_68"])
    fl = load("flow_baseline")
    if fl is not None:  # docs/flow_baseline_protocol.md: setting chosen on M0--M2
        add("Optical flow (RAFT~\\cite{teed2020raft})", fl[fl.method != "nearest"])
    ot = load("ot_baseline")
    if ot is not None:
        add("Unbalanced OT~\\cite{schiebinger2019optimal}", ot[ot.method == "ot_uot_radial"])
    cpd, cdev = load("cpd_baseline"), load("cpd_baseline_dev")
    if cpd is not None and cdev is not None:
        # variant (alignment) chosen by its best dev score, never by test
        sc = cdev.groupby(["method", "H"]).accuracy.mean().groupby("method").mean()
        pick = "_".join(sc.idxmax().split("_")[:2])
        add("Non-rigid CPD~\\cite{myronenko2010cpd}", cpd[cpd.method == pick])
    lomo = load("leave_one_movie_multihorizon_all")
    if lomo is not None:
        m = lomo[(lomo.method == "multi_horizon") & lomo.held_out_movie.isin(["M3", "M4"])].copy()
        m["accuracy"] = m.n_correct / m.n_scored
        add("Family-centre regression (GBDT)", m.rename(columns={"horizon_frames": "H"}))
    if tr is not None:
        add("Trackastra~\\cite{gallusser2024trackastra}, greedy", tr[tr.method == "trackastra_greedy"])
        add("Trackastra~\\cite{gallusser2024trackastra}, argmax", tr[tr.method == "trackastra_argmax"])
    ft = load("trackastra_ft")
    if ft is not None:
        # fine-tuned on M0/M1 at each gap; decoding (argmax) and checkpoint chosen on M2 (docs/trackastra_finetune_protocol.md)
        add("Trackastra, fine-tuned per gap", ft[ft.method == "trackastra_ft_argmax"])
        mh = load("trackastra_ft_multi")
        if mh is not None:
            # one model over five gaps (docs/matched_trackastra_protocol.md); per gap, the variant better on M2 validation
            add("Trackastra, fine-tuned over gaps", mh[mh.method == "trackastra_ft_argmax"])
            sel = pd.concat([ft[(ft.method == "trackastra_ft_argmax") & ft.H.isin([33, 66])],
                             mh[(mh.method == "trackastra_ft_argmax") & (mh.H == 132)]])
            add("Trackastra, fine-tuned$^\\dagger$", sel)
    for tag, name in (("raft_v1", "\\ours, rot.\\ aug."), ("raft_v1_norot", "\\ours, no rot."),
                      ("raft_v2_darcy_30k", "\\ours, rot.\\ aug. + Darcy init")):
        d = load(f"raft_eval_{tag}")
        add(name, None if d is None else d[d.method == f"{tag}_argmax"])

    # full model, decoder and oracle: mean over the training seeds of v3 (std kept in SEED_STD)
    def add_seeds(name, frames, col="accuracy"):
        per = []
        for f in frames:
            add(name, f, col=col)
            per.append(np.array(rows[name][0] + rows[name][1]))
        per = np.array(per)
        rows[name] = (list(per.mean(0)[:3]), list(per.mean(0)[3:]))
        SEED_STD[name] = per.std(0, ddof=1) if len(per) > 1 else np.zeros(6)

    tags = [t for t in SEEDS if load(f"raft_eval_{t}") is not None]
    dm = ("dm_darcy", "dm_darcy_s1", "dm_darcy_s2")  # same network, centre fixed (docs/directmatch_protocol.md)
    if all(load(f"raft_eval_{t}") is not None for t in dm):
        add_seeds("Direct matcher (\\ours, fixed centre)", [load(f"raft_eval_{t}").query("method == @t + '_argmax'") for t in dm])
    add_seeds("\\ours, full", [load(f"raft_eval_{t}").query("method == @t + '_argmax'") for t in tags])
    m01 = ("raft_m01_20k", "raft_m01_20k_s1", "raft_m01_20k_s2")  # docs/matched_trackastra_protocol.md
    if all(load(f"raft_eval_{t}") is not None for t in m01):
        add_seeds("\\ours, full recipe, M0/M1 only, 20k steps", [load(f"raft_eval_{t}").query("method == @t + '_argmax'") for t in m01])
    dec = [load(f"image_partition_decode_{t}") for t in tags]
    add_seeds("\\ours, full + genealogy decoder", dec, col="mf_img2")
    ens = load("image_partition_decode_" + "+".join(SEEDS))
    if ens is not None:
        # probabilities of the seeds averaged, then decoded as above (beta = 2 as for single models)
        add("\\ours, 3-seed ensemble", ens, col="argmax")
        add("\\ours, ensemble + decoder", ens, col="mf_img2")
        # fully annotation-free: v3 and genealogy both trained on chained pseudo-lineage (docs/pseudo_genealogy_protocol.md)
        free = load("image_partition_decode_" + "+".join(PSEUDO_SEEDS) + "_pseudogen")
        if free is not None:
            add("\\ours, ensemble + decoder, no annotation", free, col="mf_img2")
        add("\\textcolor{gray}{Oracle family partition}", ens, col="oracle_partition")
    else:
        add_seeds("\\textcolor{gray}{Oracle family partition}", dec, col="oracle_partition")
    return rows


OIAM_COMPACT_DROP = ("Trackastra~\\cite{gallusser2024trackastra}, greedy", "Trackastra, fine-tuned per gap",
                     "Trackastra, fine-tuned over gaps", "\\ours, full recipe, M0/M1 only, 20k steps", "\\ours, rot.\\ aug. + Darcy init",
                     "\\ours, full + genealogy decoder", "\\ours, 3-seed ensemble")


def table_oiam(full=False):
    """Main text: compact rows; supplement (full=True): every variant."""
    rows = oiam_rows()
    if not full:
        rows = {k: v for k, v in rows.items() if k not in OIAM_COMPACT_DROP}
    free_note = ("no annotation: both networks trained on self-labelled lineage of the unannotated M0--M2; "
                 if any("no annotation" in k for k in rows) else "")
    vals = {k: v for k, v in rows.items() if v is not None and "Oracle" not in k}
    best = [max(v[0][i] for v in vals.values()) for i in range(3)]
    bestd = [max(v[1][i] for v in vals.values()) for i in range(3)]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{All windows} & \multicolumn{3}{c}{Dense ($N_a{>}200$)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for k, v in rows.items():
        if "Oracle" in k:
            lines.append(r"\midrule")
        if v is None:
            lines.append(k + " & " + " & ".join([r"\TODO{}"] * 6) + r" \\")
            continue
        a, dn = v
        cells = [fmt(a[i], abs(a[i] - best[i]) < 5e-4 and "Oracle" not in k) for i in range(3)] + \
                [fmt(dn[i], abs(dn[i] - bestd[i]) < 5e-4 and "Oracle" not in k) for i in range(3)]
        lines.append(k + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{In-domain results on OIAM M3/M4} (trained on M0--M2). Accuracy at gaps of 33/66/132 minutes. "
              r"Trackastra receives images and ground-truth masks of both snapshots. $^\dagger$Fine-tuned on M0/M1 lineage, one model per gap or one over five gaps, whichever is better on M2 (per gap at 33/66, over gaps at 132 minutes; checkpoint and decoding also chosen on M2). "
              r"M3/M4 were inspected during development (leave-one-movie-out: supplement). "
              + rf"Direct matcher: the same network with the family centre frozen at its initial position. Rot.\ aug./no rot.: 30k steps, radial initialisation, with/without rotation augmentation; full: see text. Direct matcher and full" + (" (and its decoder row)" if full else "") + rf": mean of {sum(load(f'raft_eval_{t}') is not None for t in SEEDS)} seeds (std $\le${max(v.max() for v in SEED_STD.values()):.3f}); ensemble rows average the seeds' probabilities; {free_note}oracle: ensemble; other learned rows: one seed.}}",
              ("\\label{tab:oiamfull}" if full else "\\label{tab:oiam}"), r"\end{table}"]
    return "\n".join(lines)


def table_lomo():
    """GBDT, the 30k no-rotation model and (docs/full_lomo_protocol.md) the full model, leave-one-movie-out."""
    folds = ("M0", "M1", "M2", "M3", "M4")
    old = load("leave_one_movie_multihorizon_all")
    models = [("Family-centre regression", None), ("\\ours, no rot., 30k", "raft_lomo_{}"), ("\\ours, full", "raft_v3_lomo_{}")]
    if any(load("raft_eval_" + models[2][1].format(k)) is None for k in folds):
        models = models[:2]
    vals = {}
    for k in folds:
        o = old[(old.method == "multi_horizon") & (old.held_out_movie == k)].copy()
        o["acc"] = o.n_correct / o.n_scored
        row = [o[o.horizon_frames == h].acc.mean() for h in H]
        for _, pat in models[1:]:
            t = pat.format(k)
            d = load(f"raft_eval_{t}")
            row += [d[(d.method == f"{t}_argmax") & (d.H == h)].accuracy.mean() for h in H]
        vals[k] = row
    vals["Mean"] = list(np.mean([vals[k] for k in folds], 0))
    n = len(models)
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "ccc" * n + "}", r"\toprule",
             "& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{m}}}" for m, _ in models) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n)),
             "Held out & " + " & ".join(["33 & 66 & 132"] * n) + r" \\", r"\midrule"]
    for k, row in vals.items():
        if k == "Mean":
            lines.append(r"\midrule")
        cells = [fmt(row[i], row[i] >= max(row[g::3][j] for j in range(n)) - 5e-4) for i in range(3 * n) for g in [i % 3]]
        lines.append(k + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Leave-one-movie-out on OIAM.} Each row trains on the other four movies and tests on all windows of the held-out one. "
              + (r"\ours, full: the configuration of the main results (80k steps, Darcy initialisation, consistency term, no rotation), one seed, fixed before running (protocol in the ledger). " if n == 3 else "")
              + r"The five movies come from one experiment, so folds measure sensitivity to the movie, not independent replication. Bold: the best model per gap.}",
              r"\label{tab:lomo}", r"\end{table}"]
    return "\n".join(lines)

EXT = [("FAST", "FAST"), ("recA", "recA"), ("rpsM", "rpsM"), ("OIAM", "OIAM"), ("T98G", "T98G"), ("LFCT", "LFCT")]


def table_external():
    d = load("raft_external_v1")
    seeds = load("raft_external_v3")
    if seeds is not None:
        # RodSim column: per-window mean of the three training seeds (seed 0 in v1, seeds 1/2 in v3)
        key = ["dataset", "sequence", "window_id", "gap_minutes"]
        sm = pd.concat([d[d.method == "raft_sim_loc_argmax"],
                        seeds[seeds.method.isin(["raft_sim_loc_s1_argmax", "raft_sim_loc_s2_argmax"])]])
        sm = sm.groupby(key, as_index=False).accuracy.mean().assign(method="raft_sim_loc_argmax")
        d = pd.concat([d[d.method != "raft_sim_loc_argmax"], sm], ignore_index=True)
    base = ["nearest_anchor", "radial_expansion", "centroid_translation"]
    # OT, CPD, Darcy and forward simulation on the same windows (docs/external_baselines_protocol.md)
    other = ["ot_uot_radial", "ot_uot_none", "ot_exact_none", "cpd_radial", "darcy_68", "fsa_pos"]
    ext = load("external_baselines")
    if ext is not None:
        d = pd.concat([d, ext[ext.method.isin(other + ["radial_sector"])]], ignore_index=True)
    names = {"nearest_anchor": "Nearest", "radial_expansion": "Radial", "centroid_translation": "Transl.",
             "best_other": "Other$^*$",
             "raft_v1_loc_argmax": "OIAM", "raft_sim_loc_argmax": "\\rodsim"}
    cols = base + (["best_other"] if ext is not None else []) + ["raft_v1_loc_argmax", "raft_sim_loc_argmax"]
    ng = len(cols) - 2
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "c" * len(cols) + "c}", r"\toprule",
             rf"& \multicolumn{{{ng}}}{{c}}{{Baselines}} & \multicolumn{{2}}{{c}}{{\ours trained on}} & Seq. \\",
             rf"\cmidrule(lr){{2-{ng + 1}}}\cmidrule(lr){{{ng + 2}-{ng + 3}}}",
             "Dataset & " + " & ".join(names[c] for c in cols) + r" & wins \\", r"\midrule"]
    for key, label in EXT:
        x = d[d.dataset == key]
        acc = {c: x[x.method == c].groupby("gap_minutes").accuracy.mean().mean() for c in cols if c != "best_other"}
        if ext is not None:
            ov = {c: x[x.method == c].groupby("gap_minutes").accuracy.mean().mean() for c in other}
            win = max(ov, key=ov.get)
            acc["best_other"] = ov[win]
        best = max(acc.values())
        s = x.pivot_table(index="sequence", columns="method", values="accuracy")
        allbase = base + (other + ["radial_sector"] if ext is not None else [])
        wins = int((s["raft_sim_loc_argmax"] > s[allbase].max(axis=1)).sum())
        tag = {"ot": "o", "cp": "c", "da": "d", "fs": "f"}
        cell = {c: fmt(acc[c], abs(acc[c] - best) < 5e-4) + (rf"$^{{\mathrm{{{tag[win[:2]]}}}}}$" if c == "best_other" else "")
                for c in cols}
        lines.append(label + " & " + " & ".join(cell[c] for c in cols) +
                     f" & {wins}/{len(s)}" + r" \\")
    seed_note = " \\rodsim-trained: mean of three training seeds (per window); OIAM-trained: one seed, three movies, with rotation augmentation." if seeds is not None else ""
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Zero-shot transfer.} Accuracy averaged over the three gaps of each dataset. "
              r"Except for the OIAM-trained model on OIAM, no model saw labels from these datasets; the \rodsim model saw no real data at all. "
              r"$^*$Best per dataset, in hindsight, of OT (o; three settings), CPD (c), Darcy advection (d) "
              r"and \rodsim forward simulation (f), all fixed on OIAM M0--M2. "
              r"\ours uses positions only. Last column: sequences on which \rodsim-trained \ours beats the best of all ten baselines (the columns, radial sector and the six behind Other; mean over gaps)." +
              seed_note + "}", r"\label{tab:external}", r"\end{table}"]
    return "\n".join(lines)


def table_genealogy():
    real = load("snapshot_genealogy_eval")
    sim = load("sim_genealogy_eval")
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llccc}", r"\toprule", r"Training data & Input & 33 & 66 & 132 \\", r"\midrule"]
    if real is not None:
        lines.append("& Base rate & " + " & ".join(
            fmt(real[(real.H == h) & (real.model == "image")].pos_rate.mean()) for h in H) + r" \\")
    rows = [("OIAM M0--M2", real, "geometry", "Geometry"), ("OIAM M0--M2", real, "image", "Image + geometry"),
            ("\\rodsim", sim, "sim_geometry", "Geometry"), ("\\rodsim", sim, "sim_image", "Image + geometry"),
            ("\\rodsim + division modes", load("sim_genealogy_v2_v2_eval"), "simv2_v2_geometry", "Geometry"),
            ("\\rodsim + division modes", load("sim_genealogy_v2_v2_eval"), "simv2_v2_image", "Image + geometry")]
    for label, df, m, inp in rows:
        v = [None] * 3 if df is None else [df[(df.H == h) & (df.model == m)].ap.mean() for h in H]
        lines.append(f"{label} & {inp} & " + " & ".join(fmt(x) for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Genealogy is visible in one snapshot.} Average precision for ``same ancestor $H$ minutes ago'' "
              r"over 8-nearest-neighbour pairs in single OIAM M3/M4 frames. \rodsim rows use no real label; "
              r"division-modes rows also render lineage (inherited intensity, septum constriction; alone: AP .110/.166/.219).}", r"\label{tab:genealogy}", r"\end{table}"]
    return "\n".join(lines)


def table_decoder():
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"& & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{Dense} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Model & Decoder & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for tag, name in (("raft_v1_norot", "No rot."), ("raft_v3", "Full")):
        d = load(f"image_partition_decode_{tag}")
        if d is None:
            continue
        d["dense"] = d.n_anchors > 200
        for col, dec in (("argmax", "argmax"), ("mf_img1", "Potts $\\beta{=}1$"), ("mf_img2", "Potts $\\beta{=}2$"),
                         ("mf_img4", "Potts $\\beta{=}4$"), ("img0.3_free", "Cluster + argmax"),
                         ("oracle_partition", "\\textcolor{gray}{Oracle partition}")):
            v = [d[d.H == h][col].mean() for h in H] + [d[(d.H == h) & d.dense][col].mean() for h in H]
            lines.append(f"{name if col == 'argmax' else ''} & {dec} & " + " & ".join(fmt(x) for x in v) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}",
              r"\caption{\textbf{Partition-aware decoding} with single-snapshot genealogy on OIAM M3/M4. "
              r"$\beta$ was not tuned on held-out data; all values are shown; the Darcy-initialised model behaves alike. Cluster + argmax: average-linkage clusters of the genealogy affinities cut at 0.3, each cluster sent to the ancestor with the largest summed log-probability. Full is training seed 0; the main results table applies Potts $\beta{=}2$ to the probability ensemble of three seeds. " + SKIP_NOTE + "}", r"\label{tab:decoder}", r"\end{table}"]
    return "\n".join(lines)


def table_labelfree():
    """Sim-trained LineageRAFT decoded with genealogy from different sources (Potts beta=1)."""
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccc}", r"\toprule", r"Genealogy in the decoder & 33 & 66 & 132 \\", r"\midrule"]
    base = "image_partition_decode_raft_sim_loc"
    for suffix, name, col in (("", "none (argmax)", "argmax"),
                              ("_simgen", "\\rodsim", "mf_img1"),
                              ("_simgen_sim_genealogy_v2_v2", "\\rodsim + division modes", "mf_img1"),
                              ("", "OIAM M0--M2 labels", "mf_img1"),
                              ("", "\\textcolor{gray}{oracle partition}", "oracle_partition")):
        d = load(base + suffix)
        v = [None] * 3 if d is None else [d[d.H == h][col].mean() for h in H]
        lines.append(name + " & " + " & ".join(fmt(x) for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{A label-free pipeline.} \ours trained only on \rodsim, decoded with single-image "
              r"genealogy from different sources (Potts, $\beta{=}1$); OIAM M3/M4 accuracy. The first three rows use no real label. " + SKIP_NOTE + "}",
              r"\label{tab:labelfree}", r"\end{table}"]
    return "\n".join(lines)


def table_ablation():
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccc}", r"\toprule", r"Simulator variant & FAST & recA & rpsM & OIAM \\", r"\midrule"]
    d1, d2, d3 = load("raft_external_v1"), load("raft_external_simablate"), load("raft_external_simv2")
    for tag, name, src in (("raft_sim_loc", "Full \\rodsim (256 colonies)", d1),
                           ("raft_sim_fixedphys", "No physics randomisation", d2),
                           ("raft_sim_fullfov", "No field-of-view crop", d2),
                           ("raft_sim_n64", "64 colonies", d2), ("raft_sim_n16", "16 colonies", d2),
                           ("raft_sim_v2", "+ division modes", d3)):
        v = []
        for key in ("FAST", "recA", "rpsM", "OIAM"):
            if src is None:
                v.append(None)
                continue
            x = src[(src.dataset == key) & (src.method == f"{tag}_argmax")]
            v.append(x.groupby("gap_minutes").accuracy.mean().mean() if len(x) else None)
        lines.append(name + " & " + " & ".join(fmt(x) for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Simulator ablations} (zero-shot accuracy, mean over gaps; every row is one training seed, seed 0, so it differs from the three-seed means of the zero-shot table).}",
              r"\label{tab:simablate}", r"\end{table}"]
    return "\n".join(lines)


def table_blind():
    """Both pre-registered blind tests (docs/blind_test_protocol.md, docs/blind_test2_protocol.md)."""
    geo = ("nearest_anchor", "centroid_translation", "radial_expansion", "radial_sector")
    sims = ("raft_sim_loc", "raft_sim_loc_s1", "raft_sim_loc_s2")

    def seq_table(d):
        return d.groupby(["dataset", "sequence", "gap_minutes", "method"]).accuracy.mean() \
            .groupby(["dataset", "sequence", "method"]).mean().unstack("method")

    cols = {}
    a, b, c, t2 = (load(n) for n in ("blind_external_test", "blind_percolony_posthoc",
                                     "label_free_selection", "blind_external_test2"))
    t3 = load("self_label_adapt")
    adapt = seq_table(t3) if t3 is not None else None
    if a is not None and b is not None and c is not None:
        s1 = seq_table(pd.concat([a, b]).replace({"method": {f"{t}_argmax": t for t in sims}}))
        sv = c[c.method.isin(["nearest_anchor", "raft_col"]) & (c.dataset != "OIAM")]
        key = ["dataset", "sequence", "window_id", "gap_minutes"]
        sv = sv.loc[sv.sort_values("size_cv").groupby(key).coverage.idxmax()].assign(method="sv")
        s1 = s1.join(seq_table(sv))
        s1["col"] = s1[[f"col_{t}" for t in sims]].mean(axis=1)
        s1["whole"] = s1[list(sims)].mean(axis=1)
        for ds in ("TrackRefiner", "TrackMate"):
            cols[ds] = s1.loc[ds]
    if t2 is not None:
        s2 = seq_table(t2)
        s2["col"] = s2[[f"col_{t}" for t in sims]].mean(axis=1)
        s2["whole"] = s2[list(sims)].mean(axis=1)
        s2["sv"] = s2[[f"sv_{t}" for t in sims]].mean(axis=1)
        for ds in ("LYN", "BSUB"):
            cols[ds] = s2.loc[ds]
    order = ["TrackRefiner", "TrackMate", "LYN", "BSUB"]

    def val(ds, col):
        if ds not in cols:
            return None
        g = cols[ds]
        if col == "best_geo":
            return g[[k for k in g if any(x in k for x in geo)]].mean().max()
        return g[col].mean() if col in g else None

    def wins(ds, g=None, col="sv"):
        if g is None:
            if ds not in cols:
                return r"\TODO{}"
            g = cols[ds]
        best = g[[k for k in g if any(x in k for x in geo)]].max(axis=1)
        return f"{int((g[col] > best).sum())}/{len(g)}"

    def aval(ds, col):
        return None if adapt is None else adapt.loc[ds][col].mean()

    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Blind test 1} & \multicolumn{2}{c}{Blind test 2} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & TrackRefiner & TrackMate & LYN (yeast) & \emph{B.\ subtilis} \\", r"\midrule"]
    for name, col, dag in (("Nearest ancestor", "nearest_anchor", False),
                           ("Best of 8 geometric", "best_geo", False),
                           ("\\ours, whole field", "whole", False),
                           ("\\ours, per colony", "col", True),
                           ("\\ours-SV (label-free selection)", "sv", True)):
        v = [val(ds, col) for ds in order]
        bestv = max(x for x in v if x is not None) if any(x is not None for x in v) else None
        cells = [fmt(x) + (r"$^\dagger$" if dag and k < 2 and x is not None else "") for k, x in enumerate(v)]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += ["SV wins / sequences & " + " & ".join(wins(ds) + (r"$^\dagger$" if k < 2 else "") for k, ds in enumerate(order)) + r" \\",
              r"\midrule",
              r"\multicolumn{5}{l}{\emph{+ self-labelling on the other fold's unannotated sequences (protocol 3)}} \\"]
    for name, col in (("\\ours, whole field", "adapt"), ("\\ours, per colony", "col_adapt"), ("\\ours-SV", "sv_adapt")):
        lines.append(name + " & " + " & ".join(fmt(aval(ds, col)) for ds in order) + r" \\")
    lines += ["SV wins / sequences & " + " & ".join(
                  wins(ds, adapt.loc[ds], "sv_adapt") if adapt is not None else r"\TODO{}" for ds in order) + r" \\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Pre-registered blind tests and self-labelled adaptation.} Accuracy averaged over windows, gaps (30/60/120 min) "
              r"and sequences; \rodsim-trained, positions only. "
              r"Test 1 (11 sequences) was run first and failed; $^\dagger$per-colony processing and SV were designed after "
              r"seeing it, so these test-1 entries are post hoc (SV there uses one seed). Test 2 (6 yeast colonies, "
              r"44 \emph{B.\ subtilis} sequences) was fixed before its data were downloaded and run once. "
              r"Best of 8 geometric: nearest, centroid translation, radial expansion and radial sector (the ancestor in the most similar direction from the colony centroid), each on the whole field "
              r"or per colony, best chosen per source after the run. Protocol 3 (not blind: same windows, method "
              r"fixed before running) fine-tunes the \rodsim model on chained pseudo-lineage of the other half of each "
              r"source's sequences, with no ground-truth lineage.}",
              r"\label{tab:blind}", r"\end{table}"]
    return "\n".join(lines)


def table_selflabel():
    """Self-labelling on OIAM M3/M4: chained pseudo-lineage of M0-M2 vs ground-truth lineage."""
    rows = [("Positions only, seed 0", None, None),
            ("\\rodsim only", "raft_sim_loc", "--"),
            ("\\quad + fine-tune, true lineage", "raft_sim_gtft", "M0--M2"),
            ("\\quad + fine-tune, self-labelled", "raft_sim_pseudo", "none"),
            ("Full model (shape, Darcy), mean of three seeds", None, None),
            ("True lineage", SEEDS, "M0--M2"),
            ("Self-labelled", PSEUDO_SEEDS, "none"),
            ("\\quad without consistency term (post hoc)", ("raft_pseudo", "raft_pseudo_s1", "raft_pseudo_s2"), "none")]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & Lineage & \multicolumn{3}{c}{All windows} & \multicolumn{3}{c}{Dense ($>$200)} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Training & labels & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for name, tag, lab in rows:
        if tag is None:
            lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{name}}}}} \\")
            continue
        vs = []
        for t in ((tag,) if isinstance(tag, str) else tag):
            d = load(f"raft_eval_{t}")
            if d is None:
                continue
            d = d[d.method == f"{t}_argmax"].assign(dense=lambda x: x.n_anchors > 200)
            vs.append([d[d.H == h].accuracy.mean() for h in H] + [d[(d.H == h) & d.dense].accuracy.mean() for h in H])
        v = list(np.mean(vs, 0)) if vs else [None] * 6
        lines.append(f"{name} & {lab} & " + " & ".join(fmt(x) for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Self-labelling replaces lineage annotation} (OIAM M3/M4, accuracy, argmax). "
              r"Self-labelled models are trained on lineage obtained by chaining one-frame links through the "
              r"\emph{unannotated} M0--M2 time-lapses (\cref{sec:selflabel}); fine-tuning runs 6k steps from the "
              r"\rodsim model, full models 80k steps from scratch (seed std $\le$0.013).}",
             r"\label{tab:selflabel}", r"\end{table}"]
    return "\n".join(lines)


def table_core():
    """Core ablation of family registration (docs/core_ablation_protocol.md, docs/directmatch_protocol.md)."""
    rows = [("30k steps, seed 0 (core ablation)", None),
            ("$T{=}4$, moving centre (reference)", "raft_abl_T4"),
            ("$T{=}2$", "raft_abl_T2"),
            ("$T{=}1$", "raft_abl_T1"),
            ("$T{=}0$: learned pairwise matcher", "raft_abl_T0"),
            ("$T{=}4$, fixed centre (direct matcher)", "raft_abl_fix"),
            ("80k steps, mean of three seeds (matched direct matcher)", None),
            ("\\ours, Darcy init.", SEEDS),
            ("Direct matcher, Darcy init.", ("dm_darcy", "dm_darcy_s1", "dm_darcy_s2")),
            ("\\ours, radial init.", ("raft_v3_nodarcy", "raft_v3_nodarcy_s1", "raft_v3_nodarcy_s2")),
            ("Direct matcher, radial init.", ("dm_radial", "dm_radial_s1", "dm_radial_s2")),
            ("Direct matcher, 96 candidates", ("dm_k96", "dm_k96_s1", "dm_k96_s2")),
            ("Direct matcher + centre loss", ("dm_aux", "dm_aux_s1", "dm_aux_s2")),
            ("\\ours, $T{=}1$", ("raft_T1", "raft_T1_s1", "raft_T1_s2")),
            ("80k steps, seed 0", None),
            ("\\ours without consistency term", "raft_v3_noconsist"),
            ("\\ours, $T{=}8$", "raft_v4_it8")]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{All windows} & \multicolumn{3}{c}{Dense ($>$200)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for name, tag in rows:
        if tag is None:
            lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{name}}}}} \\")
            continue
        vs = []
        for t in ((tag,) if isinstance(tag, str) else tag):
            d = load(f"raft_eval_{t}")
            if d is None:
                vs = []
                break
            d = d[d.method == f"{t}_argmax"].assign(dense=lambda x: x.n_anchors > 200)
            vs.append([d[d.H == h].accuracy.mean() for h in H] + [d[(d.H == h) & d.dense].accuracy.mean() for h in H])
        v = list(np.mean(vs, 0)) if vs else [None] * 6
        lines.append(f"{name} & " + " & ".join(fmt(x) for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{What family registration needs} (OIAM M3/M4, accuracy, argmax). "
              r"All rows use shape features, no rotation augmentation and the consistency term; Darcy initialisation unless stated. "
              r"$T$: rounds of cross-snapshot attention and centre update. With $T{=}0$ each snapshot is encoded on its own and "
              r"targets are scored against the initial anchor positions, a learned pairwise matcher. The direct matcher is the same "
              r"network with the same 2.9M parameters, data, losses and schedule, except that the family centre never moves and has no loss: "
              r"targets cross-attend to, and are scored against, candidate ancestors at their initial positions. "
              r"Controls: the direct matcher with 96 instead of 24 candidate ancestors per target, or with the centre head "
              r"trained by the centre loss but never used. "
              r"All three protocols were fixed before training and run once.}",
              r"\label{tab:core}", r"\end{table}"]
    return "\n".join(lines)


def table_decoder_baselines():
    """The genealogy decoder on baseline probabilities (docs/decoder_on_baselines_protocol.md)."""
    d = load("decoder_on_baselines")
    if d is None:
        return r"\TODO{decoder baselines}"
    d = d.assign(dense=d.n_anchors > 200)
    rows = (("Trackastra, fine-tuned", "trackastra_argmax", "argmax"),
            ("", "trackastra_mf2", r"+ decoder, $T$ fitted on M2"),
            ("", "trackastra_mf2_T1", r"+ decoder, $T{=}1$"),
            ("Family-centre regression", "gbdt_own", "own geometric decoder"),
            ("", "gbdt_argmax", "argmax"),
            ("", "gbdt_mf2", "+ decoder"),
            ("\\ours, full (3 seeds)", "raft_argmax", "argmax"),
            ("", "raft_mf2", "+ decoder"),
            ("\\ours, 3-seed ensemble", "raft_ens_mf2", "+ decoder"))
    a = d.pivot_table(index="method", columns="H", values="accuracy")
    b = d[d.dense].pivot_table(index="method", columns="H", values="accuracy")
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{All windows} & \multicolumn{3}{c}{Dense ($>$200)} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}", r"Base & Decoding & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for name, key, dec in rows:
        lines.append(f"{name} & {dec} & " + " & ".join(fmt(t[h][key], t[h][key] >= t[h].max() - 5e-4)
                                                        for t in (a, b) for h in (33, 66, 132)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\caption{\textbf{The genealogy decoder on baseline probabilities} (OIAM M3/M4, decoder windows; Potts $\beta{=}2$ "
              r"on the same image affinities; protocol fixed before running). Trackastra: row-normalised association weights; "
              r"GBDT: softmax of its own cost at its decoder temperature, and its own collective decoder with geometric "
              r"affinities and capacity (the main-table setting).}", r"\label{tab:decbase}", r"\end{table}"]
    return "\n".join(lines)


def table_block_bootstrap():
    """Window \vs time-block bootstrap CIs (docs/block_bootstrap_protocol.md)."""
    d = load("block_bootstrap")
    if d is None:
        return r"\TODO{block bootstrap}"
    names = {"C1": r"Full $-$ Trackastra, fine-tuned", "C2": r"Full $-$ direct matcher", "C3": r"Full $-$ regressor (GBDT)",
             "C4": r"Ens.\ + dec.\ $-$ Trackastra, fine-tuned", "C5": r"No annotation $-$ Trackastra, fine-tuned",
             "C6a": r"Full + dec.\ $-$ Trackastra + dec.\ (fitted $T$)", "C6b": r"Full + dec.\ $-$ Trackastra + dec.\ ($T{=}1$)"}
    c = lambda lo, hi: (r"\textbf{" if lo > 0 else "{") + f"{lo:+.3f}, {hi:+.3f}" + "}"
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
             r" & \multicolumn{3}{c}{33 min} & \multicolumn{3}{c}{66 min} & \multicolumn{3}{c}{132 min} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Comparison" + r" & mean & windows & blocks of 4" * 3 + r" \\", r"\midrule"]
    for key, name in names.items():
        x = d[d.comparison.str.startswith(key + " ")].set_index("H")
        lines.append(name + "".join(f" & {x.loc[h, 'mean']:+.3f} & {c(x.loc[h, 'window_lo'], x.loc[h, 'window_hi'])}"
                                    f" & {c(x.loc[h, 'block4_lo'], x.loc[h, 'block4_hi'])}" for h in (33, 66, 132)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Window \vs time-block bootstrap} (OIAM M3/M4; 95\% CIs of the mean paired difference in accuracy, "
              r"bold when above zero). Blocks: four consecutive windows (132 minutes of anchor times) resampled within each movie. "
              r"Full: mean of three seeds; ensemble, decoder and no-annotation rows on the decoder windows.}",
              r"\label{tab:blockboot}", r"\end{table*}"]
    return "\n".join(lines)


def table_partition():
    """Family-partition metrics (experiments/partition_metrics.py): grouping regardless of which ancestor is named."""
    d = load("partition_metrics")
    if d is None:
        return r"\TODO{partition}"
    rows = (("Nearest ancestor", "nearest"), ("Family-centre regression (GBDT)", "gbdt"),
            ("\\ours (mean of three seeds)", "single_argmax"), ("\\quad + genealogy decoder", "single_mf2"),
            ("\\ours, 3-seed ensemble", "ens_argmax"), ("\\quad + genealogy decoder", "ens_mf2"),
            ("\\quad + decoder, no annotation", "ens_noann_mf2"))
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lccccccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{Pair family F1} & \multicolumn{3}{c}{F1, dense} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Method & 33 & 66 & 132 & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for name, m in rows:
        g = d[d.method == m]
        v = [g[g.H == h].accuracy.mean() for h in H] + [g[g.H == h].pair_f1.mean() for h in H] + \
            [g[(g.H == h) & g.dense].pair_f1.mean() for h in H]
        lines.append(f"{name} & " + " & ".join(fmt(x) for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Grouping, scored directly} (OIAM M3/M4, windows with at least nine targets). "
              r"Pair family F1 counts pairs of scored targets that share the true ancestor and are given the same ancestor, "
              r"whichever it is; the adjusted Rand index agrees within 0.03. The oracle partition scores 1 by construction. "
              r"GBDT is retrained on M0--M2 here, so its accuracy differs slightly from the main table.}",
              r"\label{tab:partition}", r"\end{table}"]
    return "\n".join(lines)


def table_decomp():
    """Conditional error decomposition (experiments/error_decomposition.py, docs/error_decomposition_protocol.md)."""
    d = load("error_decomposition")
    if d is None:
        return r"\TODO{decomp}"
    d["dense"] = d.n_anchors > 200
    per = d.groupby(["H", "window_id"]).mean(numeric_only=True)
    wm = pd.concat({False: per.groupby("H").mean(), True: per[per.dense > 0].groupby("H").mean()})
    rows = (("Network (ensemble argmax)", "ok_argmax"), ("+ true family centres", "ok_oracle_centre"),
            ("+ true partition", "ok_oracle_partition"), ("+ both", "ok_oracle_both"),
            ("True ancestor among the 24 candidates", "in_cand_any"))
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{All windows} & \multicolumn{3}{c}{Dense ($N_a{>}200$)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Intervention & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for name, c in rows:
        if c == "in_cand_any":
            lines.append(r"\midrule")
        v = [wm.loc[(dn, h), c] for dn in (False, True) for h in H]
        lines.append(f"{name} & " + " & ".join(fmt(x) for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Which oracle fixes long gaps?} OIAM M3/M4, three-seed ensemble of the full model, "
              r"window-mean accuracy. True centres: the final family centre of every ancestor is replaced by the centroid "
              r"of its true family and the network re-scores its candidates (features unchanged). True partition: the "
              r"descendant partition is given to the decoder. Protocol fixed before the run.}",
              r"\label{tab:decomp}", r"\end{table}"]
    return "\n".join(lines)


def table_structured():
    """Genealogy-constrained decoder, confirmatory movie M4 (docs/genealogy_constrained_protocol.md)."""
    d = load("genealogy_constrained_decode")
    if d is None:
        return r"\TODO{structured}"
    d = d[d.sequence == "M4"].assign(dense=lambda x: x.n_anchors > 200)
    rows = (("Argmax", "argmax"), ("Potts (main paper)", "potts"), ("Growth prior only", "growth_only"),
            ("Pairwise likelihood only", "pairwise_only"), ("Structured", "gc_noEM"),
            ("Structured + centre update", "gc_full"), ("\\textcolor{gray}{True family sizes}", "oracle_card"),
            ("\\textcolor{gray}{True partition}", "oracle_partition"))
    g = d.groupby(["decoder", "H"]).mean(numeric_only=True)
    gd = d[d.dense].groupby(["decoder", "H"]).mean(numeric_only=True)
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{lccccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{Dense acc.} & \multicolumn{2}{c}{Pair F1}"
             r" & Size err. \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
             r"Decoder & 66 & 132 & 66 & 132 & 66 & 132 & 132 \\", r"\midrule"]
    for name, m in rows:
        if m == "oracle_card":
            lines.append(r"\midrule")
        v = [g.loc[(m, h), "acc"] for h in (66, 132)] + [gd.loc[(m, h), "acc"] for h in (66, 132)] + \
            [g.loc[(m, h), "f1"] for h in (66, 132)]
        lines.append(f"{name} & " + " & ".join(fmt(x) for x in v) + f" & {g.loc[(m, 132), 'size_err']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{A family-size prior repairs grouping, not long-gap ancestry} (OIAM M4, the confirmatory "
              r"movie; same ensemble scores and genealogy affinities for every decoder; protocol and hyperparameters fixed "
              r"before M4 was decoded). Size err.: mean $|n_{\text{pred}}-n_{\text{true}}|$ per ancestor; truth has std "
              + f"{d[d.decoder == 'truth'].groupby('H').size_std.mean()[132]:.2f}" + r" of family sizes at 132 minutes.}",
              r"\label{tab:structured}", r"\end{table}"]
    return "\n".join(lines)


def table_regime():
    """Post hoc: moving centre minus direct matcher by crowding (experiments/regime_analysis.py)."""
    d = load("regime_analysis")
    if d is None:
        return r"\TODO{regime}"
    d = d.dropna(subset=["disp", "diff"]).copy()
    d["crowd"] = pd.cut(d.n_anchors, [0, 25, 100, 400, 1e5], labels=["c1", "c2", "c3", "c4"])
    groups = (("OIAM, trained on M0--M2", (d.regime == "in-domain")),
              ("OIAM, trained on \\rodsim", (d.regime == "zero-shot") & (d.dataset == "OIAM")),
              ("FAST/recA/rpsM, \\rodsim", (d.regime == "zero-shot") & d.dataset.isin(["FAST", "recA", "rpsM"])),
              ("T98G/LFCT, \\rodsim", (d.regime == "zero-shot") & d.dataset.isin(["T98G", "LFCT"])))
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Median} & \multicolumn{4}{c}{Margin by ancestors per window} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
             r"Test windows, training & $N_a$ & disp. & $\le$25 & 26--100 & 101--400 & $>$400 \\", r"\midrule"]
    for name, m in groups:
        s = d[m]
        cells = []
        for c in ("c1", "c2", "c3", "c4"):
            x = s[s.crowd == c]["diff"]
            cells.append(f"{x.mean():+.3f} ({len(x)})" if len(x) >= 5 else "--")
        lines.append(f"{name} & {s.n_anchors.median():.0f} & {s.disp.median():.1f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Where the moving centre helps} (post hoc). Per-window accuracy of \ours minus the direct "
              r"matcher trained identically (three seeds each; in brackets the number of windows), all gaps pooled. "
              r"Disp.: median displacement of targets from their true ancestor after removing the mean shift, in anchor "
              r"spacings. The external bacterial windows are small; on crowded OIAM windows the centre learned from "
              r"\rodsim adds little.}",
              r"\label{tab:regime}", r"\end{table}"]
    return "\n".join(lines)


def table_longgap():
    """Three attempts at long gaps, each under its protocol: coupled refinement (docs/coupled_refinement_protocol.md),
    RodSim v3 (docs/sim_v3_protocol.md), one midpoint snapshot (docs/adaptive_snapshot_protocol.md)."""
    import sys
    sys.path.insert(0, str(ROOT))
    cpl = RES / "coupled_confirm.csv"
    ada = load("adaptive_snapshot")
    if not cpl.exists() or ada is None or not (RES / "raft_external_simv3.parquet").exists():
        return r"\TODO{longgap}"
    from experiments.sim_v3_analysis import arm_means, oiam_ci
    from experiments.adaptive_snapshot import boot
    c = pd.read_csv(cpl).rename(columns={"Unnamed: 0": "key"})
    w = arm_means()

    def cell(m, lo, hi):
        return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llccc}", r"\toprule", r"Comparison & Windows & 33 min & 66 min & 132 min \\", r"\midrule"]
    for sub in ("all", "dense"):
        x = c[c.key == f"argmax|accuracy|{sub}"].set_index("H")
        lines.append((r"Coupled $-$ \ours (OIAM, 3+3 seeds)" if sub == "all" else "") + f" & {sub} & " +
                     " & ".join(cell(x.loc[h, "diff"], x.loc[h, "blk_lo"], x.loc[h, "blk_hi"]) for h in (33, 66, 132)) + r" \\")
    lines.append(r"\midrule")
    for sub, dense in (("all", False), ("dense", True)):
        x = oiam_ci(w, "v3", "v1", dense).set_index("H")
        lines.append((r"\rodsim v3 $-$ v1, zero-shot (OIAM)" if sub == "all" else "") + f" & {sub} & " +
                     " & ".join(cell(x.loc[h, "diff"], x.loc[h, "blk_lo"], x.loc[h, "blk_hi"]) for h in (33.0, 66.0, 132.0)) + r" \\")
    for sub, dense in (("dense", True),):
        x = oiam_ci(w, "v3", "v3dm", dense).set_index("H")
        lines.append(r"\rodsim v3: \ours $-$ direct matcher" + f" & {sub} & " +
                     " & ".join(cell(x.loc[h, "diff"], x.loc[h, "blk_lo"], x.loc[h, "blk_hi"]) for h in (33.0, 66.0, 132.0)) + r" \\")
    lines.append(r"\midrule")
    for sub in ("all", "dense"):
        cells = ["--"]
        for h in (66, 132):
            s = ada[ada.H == h]
            s = s[s.n_anchors > 200] if sub == "dense" else s
            m, lo, hi = boot((s.chain - s.direct).to_numpy())
            cells.append(f"{s.direct.mean():.3f}$\\to${s.chain.mean():.3f} ({m:+.3f} [{lo:+.3f}, {hi:+.3f}])")
        lines.append((r"Midpoint snapshot, ensemble" if sub == "all" else "") + f" & {sub} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Three attempts at long gaps}, each with hypotheses fixed before running (\cref{tab:ledger}). "
              r"Window-mean argmax accuracy differences with block-bootstrap 95\% CIs (window bootstrap for the snapshot rows); "
              r"dense: $>$200 ancestors. Coupled: E/M competition of ancestors for targets inside each refinement round, "
              r"same recipe as \ours otherwise. \rodsim v3: colonies of 4{,}000--9{,}000 instead of 900--2{,}200 cells, "
              r"all arms at learning rate $2.5{\times}10^{-4}$. Midpoint snapshot: assignment composed through one extra "
              r"snapshot halfway through the window.}",
              r"\label{tab:longgap}", r"\end{table*}"]
    return "\n".join(lines)


def table_downstream():
    """Biology read from predicted ancestry: family size per ancestor and size control."""
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.downstream_growth import summary
    if load("downstream_growth") is None:
        return r"\TODO{downstream}"
    r = summary()
    rows = [("gt", "Ground truth"), ("nearest", "Nearest ancestor"), ("raft_sim_loc", "\\rodsim only"),
            ("raft_pseudo", "\\ours, self-labelled"), ("raft_v3", "\\ours")]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Size control (33 min)} & \multicolumn{3}{c}{CV of family size} & $r(\hat n,n)$ \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
             r"Method & $\rho$ & long & short & 33 & 66 & 132 & 66 \\", r"\midrule"]
    for key, name in rows:
        a = r.loc[(33, key)]
        v = [a.rho_len, a.p_div_long, a.p_div_short] + [r.loc[(h, key)].cv for h in H] + [r.loc[(66, key)].pearson]
        lines.append(f"{name} & " + " & ".join(fmt(float(x)) for x in v) + r" \\")
        if key == "gt":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Biology read from two snapshots} (OIAM M3/M4, 32 random windows per gap and movie). "
              r"Family size $n$ is the number of descendants assigned to an ancestor. Size control: Spearman $\rho$ between "
              r"ancestor length (ranked within window) and $n$, and the fraction of the longest and shortest third of ancestors "
              r"that divided within 33 minutes. CV: coefficient of variation of $n$ across ancestors (growth heterogeneity). "
              r"$r(\hat n,n)$: Pearson correlation of predicted with true family size at 66 minutes.}",
              r"\label{tab:downstream}", r"\end{table}"]
    return "\n".join(lines)


def table_bootstrap():
    """Protocols 4/5 on FAST, recA, rpsM: Hungarian vs model-chained (bootstrapped) self-labels."""
    d = load("self_label_adapt_ext2")
    if d is None:
        return r"\TODO{bootstrap}"
    seq = d.groupby(["dataset", "sequence", "gap", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    geo = ["nearest_anchor", "centroid_translation", "radial_expansion", "radial_sector"]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Dataset & Best geom. & \rodsim & + Hungarian & + bootstrap & wins \\", r"\midrule"]
    for ds in ("FAST", "recA", "rpsM"):
        g = seq.loc[ds]
        src = g.mean()
        v = [src[geo].max(), src.sim, src.adapt, src.adaptm]
        lines.append(f"{ds} & " + " & ".join(fmt(float(x), bold=(i == 3)) for i, x in enumerate(v))
                     + f" & {(g.adaptm > g.sim).sum()}/{len(g)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Temporal bootstrapping} on the microcolony datasets (accuracy, mean over colonies of the mean over gaps; "
              r"two folds per dataset, each model evaluated on the other fold). The \rodsim model is fine-tuned on pseudo-lineage "
              r"chained from positions-only Hungarian links (protocol 4) or from its own one-frame predictions (protocol 5). "
              r"Wins: colonies on which bootstrapping beats the \rodsim model.}",
              r"\label{tab:bootstrap}", r"\end{table}"]
    return "\n".join(lines)


def table_blind4():
    """Blind test 4 (docs/blind_test4_protocol.md): unused van Vliet M9 subsets, run once."""
    d = load("blind_test4")
    if d is None:
        return r"\TODO{blind4}"
    extra = load("blind4_baselines_extra")  # added after the test (docs/external_baselines_protocol.md)
    later = ["ot_uot_radial", "cpd_radial", "darcy_68", "fsa_pos"]
    if extra is not None:
        d = pd.concat([d, extra[extra.method.isin(later)][["dataset", "sequence", "gap", "window_id", "method", "accuracy"]]])
    seq = d.groupby(["dataset", "sequence", "gap", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    geo = ["nearest_anchor", "centroid_translation", "radial_expansion", "radial_sector"]
    ot = ["ot_exact", "ot_uot"] + (later if extra is not None else [])
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"Subset & $n$ & Best geom. & Best other & \rodsim & + Hung. & + boot. & \multicolumn{2}{c}{wins} \\",
             r"\cmidrule(lr){8-9}", r" & & & & & & & \rodsim & boot. \\", r"\midrule"]
    for ds in ("trpL", "metA", "pheA"):
        g = seq.loc[ds]
        src = g.mean()
        v = [src[geo].max(), src[ot].max(), src.sim, src.adapt, src.adaptm]
        best = max(range(len(v)), key=lambda i: v[i])
        lines.append(f"{ds} & {len(g)} & " + " & ".join(fmt(float(x), bold=(i == best)) for i, x in enumerate(v))
                     + f" & {(g.sim > g[geo + ot].max(axis=1)).sum()}/{len(g)}"
                     + f" & {(g.adaptm > g.sim).sum()}/{len(g)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Pre-registered blind test} on three \emph{E.~coli} subsets of~\cite{vanvliet2018spatially} in minimal medium, "
              r"never downloaded before the protocol was fixed (accuracy: mean over colonies of the mean over 30/60/120-minute gaps; run once). "
              r"Best geom.: of nearest, translation, radial expansion and radial sector; best other: of exact and unbalanced OT (tuned on OIAM) "
              r"and, added afterwards, radial-aligned OT, CPD, Darcy and forward simulation. Hung./boot.: \rodsim model fine-tuned on Hungarian-chained or bootstrapped "
              r"pseudo-lineage of the other fold. Wins: colonies where \rodsim beats all ten baselines (also the six pre-registered), and where bootstrapping beats \rodsim.}",
              r"\label{tab:blind4}", r"\end{table}"]
    return "\n".join(lines)


def table_calibration():
    """Calibration of single v3 seeds vs their ensemble (experiments/calibration.py), pooled over targets."""
    d = load("calibration")
    if d is None:
        return r"\TODO{calibration}"
    sys.path.insert(0, str(RES.parents[1]))
    from experiments.calibration import ece
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Gap & Accuracy & ECE & NLL of true ancestor \\", r"\midrule"]
    for h, g in d.groupby("H"):
        single = g[g.method != "ens3"].groupby("method")
        e = g[g.method == "ens3"]
        sv = [(x.correct.mean(), ece(x.conf.to_numpy(), x.correct.to_numpy().astype(float)),
               -np.log(np.maximum(x.p_true, 1e-9)).mean()) for _, x in single]
        ev = (e.correct.mean(), ece(e.conf.to_numpy(), e.correct.to_numpy().astype(float)),
              -np.log(np.maximum(e.p_true, 1e-9)).mean())
        cells = [f"{np.mean([v[k] for v in sv]):.3f} $\\to$ {ev[k]:.3f}" for k in range(3)]
        lines.append(f"{h} min & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Calibration} of the full model on OIAM M3/M4, pooled over scored targets: mean of three single seeds "
              r"$\to$ their probability-averaged ensemble. ECE: expected calibration error of the top choice (15 bins). "
              r"Single models are overconfident; averaging at least halves ECE at 33 and 66 minutes.}",
              r"\label{tab:calibration}", r"\end{table}"]
    return "\n".join(lines)


def table_temperature():
    """Temperature fitted on M0-M2 before the Potts decoder (experiments/temperature_decode.py)."""
    d = load("temperature_decode")
    if d is None:
        return r"\TODO{temperature}"
    d = d.assign(dense=d.n_anchors > 200)
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{All windows} & \multicolumn{3}{c}{Dense} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Probabilities $\to$ decoder & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    single = d[d.method != "ens3"]
    ens = d[d.method == "ens3"]
    for label, x, col in (("Single seed, argmax", single, "argmax"), ("Single seed", single, "mf2"),
                          ("Single seed, fitted $T$", single, "mf2_T"), ("Ensemble, argmax", ens, "argmax"),
                          ("Ensemble", ens, "mf2"), ("Ensemble, fitted $T$", ens, "mf2_T")):
        v = [x[x.H == h].groupby("method")[col].mean().mean() for h in H] + \
            [x[(x.H == h) & x.dense].groupby("method")[col].mean().mean() for h in H]
        lines.append(label + " & " + " & ".join(fmt(a) for a in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Temperature before decoding} (OIAM M3/M4; Potts $\beta{=}2$ unless argmax). "
              r"$T$ is chosen per model and gap by the likelihood of the true ancestor on M0--M2 and applied once. "
              r"Single-seed rows: mean of three seeds. Here each seed is normalised before averaging, which moves ensemble rows by at most 0.001 from the main table. " + SKIP_NOTE + "}", r"\label{tab:temperature}", r"\end{table}"]
    return "\n".join(lines)


def reliability_frames():
    """Per-window accuracy and scored-target counts of the key OIAM M3/M4 rows (for weighting and paired tests)."""
    keep = ["window_id", "H", "n_scored", "accuracy"]
    ns = load("raft_eval_raft_v3").query("method == 'raft_v3_argmax'")[["window_id", "H", "n_scored"]]
    tr, ot, cpd = load("trackastra_gap"), load("ot_baseline"), load("cpd_baseline")
    lomo = load("leave_one_movie_multihorizon_all")
    g = lomo[(lomo.method == "multi_horizon") & lomo.held_out_movie.isin(["M3", "M4"])].copy()
    g = g.assign(accuracy=g.n_correct / g.n_scored, H=g.horizon_frames)
    seeds = [load(f"raft_eval_{t}").query("method == @t + '_argmax'") for t in SEEDS]
    full = pd.concat(seeds).groupby(["window_id", "H"], as_index=False).agg(n_scored=("n_scored", "first"), accuracy=("accuracy", "mean"))
    out = {"Nearest ancestor": tr[tr.method == "nearest"][keep],
           "Unbalanced OT": ot[ot.method == "ot_uot_radial"][keep],
           "Non-rigid CPD": cpd[cpd.method == "cpd_radial"][keep] if (cpd.method == "cpd_radial").any() else cpd[cpd.method == cpd.method.iloc[0]][keep],
           "Family-centre regression (GBDT)": g[keep],
           "\\ours, full (mean of 3 seeds)": full[keep]}
    ft = load("trackastra_ft")
    if ft is not None:  # insert before the GBDT row
        items = list(out.items())
        items.insert(3, ("Trackastra, fine-tuned per gap", ft[ft.method == "trackastra_ft_argmax"][keep]))
        out = dict(items)
    for name, f, c in (("\\ours, ensemble + decoder", "image_partition_decode_" + "+".join(SEEDS), "mf_img2"),
                       ("\\ours, no annotation", "image_partition_decode_" + "+".join(PSEUDO_SEEDS) + "_pseudogen", "mf_img2"),
                       ("\\textcolor{gray}{Oracle family partition}", "image_partition_decode_" + "+".join(SEEDS), "oracle_partition")):
        d = load(f)
        if d is not None:
            out[name] = d.merge(ns, on=["window_id", "H"]).rename(columns={c: "accuracy"})[keep]
    return out


def table_reliability():
    """Window-averaged accuracy with 95% bootstrap CI over windows, cell-weighted accuracy, paired tests."""
    fr = reliability_frames()
    rng = np.random.default_rng(0)
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{Window-averaged (95\% CI)} & \multicolumn{3}{c}{Cell-weighted} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Method & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for name, d in fr.items():
        cells, cw = [], []
        for h in H:
            x = d[d.H == h]
            a, n = x.accuracy.to_numpy(), x.n_scored.to_numpy()
            bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)])
            cells.append(f"{fmt(a.mean())}{{\\scriptsize$\\pm${fmt(1.96 * bs.std())}}}")
            cw.append(fmt(np.sum(a * n) / n.sum()))
        if "Oracle" in name:
            lines.append(r"\midrule")
        lines.append(name + " & " + " & ".join(cells + cw) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\caption{\textbf{Uncertainty and weighting} on OIAM M3/M4. Left: accuracy averaged over windows (as in the main table) with the "
              r"half-width of a 95\% bootstrap interval over windows. Right: the same predictions weighted by scored targets, which counts "
              r"the dense windows more. Decoder rows use the windows with at least 9 targets.}",
              r"\label{tab:reliability}", r"\end{table}"]
    return "\n".join(lines)


def paired(a_name, b_name, h, fr=None, n_boot=4000):
    """Paired comparison over shared windows: mean difference, 95% bootstrap CI, wins/losses."""
    fr = fr or reliability_frames()
    m = fr[a_name].query("H == @h").merge(fr[b_name].query("H == @h"), on="window_id")
    d = (m.accuracy_x - m.accuracy_y).to_numpy()
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d > 0).sum()), int((d < 0).sum()), len(d)


def main():
    OUT.mkdir(exist_ok=True)
    for name, fn in (("oiam", table_oiam), ("oiam_full", lambda: table_oiam(True)), ("decoder_baselines", table_decoder_baselines), ("block_bootstrap", table_block_bootstrap), ("lomo", table_lomo), ("external", table_external),
                     ("genealogy", table_genealogy), ("decoder", table_decoder), ("simablate", table_ablation), ("labelfree", table_labelfree), ("blind", table_blind),
                     ("selflabel", table_selflabel), ("downstream", table_downstream), ("bootstrap", table_bootstrap), ("blind4", table_blind4), ("calibration", table_calibration), ("temperature", table_temperature), ("reliability", table_reliability), ("core", table_core), ("partition", table_partition),
                     ("decomp", table_decomp), ("structured", table_structured), ("regime", table_regime), ("longgap", table_longgap)):
        tex = fn()
        if "adjustbox" not in tex:
            tex = tex.replace(r"\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}", 1)
            tex = tex.replace(r"\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)
        (OUT / f"{name}.tex").write_text(tex + "\n")
        print("wrote", name)


if __name__ == "__main__":
    main()
