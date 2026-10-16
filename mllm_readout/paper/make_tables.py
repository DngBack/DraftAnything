"""Build LaTeX tables from results/ (run from mllm_readout/: python paper/make_tables.py)."""
import os
import numpy as np
import pandas as pd

R, T = "results", "paper/tables"
MODEL = {"qwen3vl2b": "Qwen3-VL-2B", "qwen3vl4b": "Qwen3-VL-4B", "qwen3vl8b": "Qwen3-VL-8B", "qwen3vl32b": "Qwen3-VL-32B",
         "internvl35_8b": "InternVL3.5-8B", "gemma3_4b": "Gemma-3-4B", "gemma3_27b": "Gemma-3-27B",
         "qwen2vl_7b": "Qwen2-VL-7B", "llava15_7b": "LLaVA-1.5-7B"}
ORDER = list(MODEL)
TASK = {"count": "Count (synthetic)", "count_real": "CountBench", "spatial": "Spatial (synthetic)",
        "spatial_real": "What'sUp", "attr": "Attribute (synthetic)", "color_cf": "Visual CounterFact"}


def decomp():
    d = pd.concat([pd.read_csv(f"{R}/g15_table.csv"), pd.read_csv(f"{R}/g2_table.csv")]).drop_duplicates("cell")
    d["model_key"] = d.cell.map(lambda c: next(m for m in sorted(MODEL, key=len, reverse=True) if c.startswith(m + "_")))
    d["task"] = [c[len(m) + 1:] for c, m in zip(d.cell, d.model_key)]
    d = d[d.n_err >= 30].copy()
    d["o"] = d.model_key.map(ORDER.index)
    d = d.sort_values(["task", "o"], key=lambda s: s.map(list(TASK).index) if s.name == "task" else s)
    rows, last = [], None
    for r in d.itertuples():
        rec = r.cal_recovers
        rec_s = "--" if pd.isna(rec) else "$>$100" if rec > 1 else f"{100 * rec:.0f}"
        rec_s = rf"\textbf{{{rec_s}}}" if rec >= 0.5 else rec_s
        task = TASK[r.task] if r.task != last else ""
        if r.task != last and last is not None:
            rows.append(r"\midrule")
        last = r.task
        rows.append(f"{task} & {MODEL[r.model_key]} & {r.n_err} & {100 * r.model:.1f} & {100 * r.calib:.1f} & {100 * r.probe:.1f} & "
                    f"{rec_s} & {100 * r.bias:.0f} & {100 * r.loss:.0f} & {100 * r.perc:.0f} \\\\")
    n_ok = int((d.cal_recovers >= 0.5).sum())
    body = "\n".join(rows)
    open(f"{T}/decomp.tex", "w").write(rf"""\begin{{table*}}[t]
\centering\small
\caption{{\textbf{{Per-error decomposition}} (5-fold cross-fitted; cells with $\geq$30 errors). Accuracies of the model, the calibrator on its own candidate logits, and the best hidden-state probe; \emph{{Rec.}} = share of the probe--model gap recovered by calibration (bold: $\geq$50\%, {n_ok}/{len(d)} cells; above 100\% when the calibrator beats the best probe, which reads one layer and position while the calibrator reads the final logits). Right: share of errors that are \bias / \loss / \perc.}}
\label{{tab:decomp}}
\begin{{tabular}}{{llrrrrrrrr}}
\toprule
Task & Model & \#err & Model & Calib & Probe & Rec.\,(\%) & \bias & \loss & \perc \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
    print("decomp:", n_ok, "/", len(d), "cells recover >= 50%")


def gencal():
    s = pd.read_csv(f"{R}/gencal_stats.csv")
    s["o"] = s.model.map(ORDER.index)
    # cells where >10% of items never reach a decision step (answer format not followed, e.g. LLaVA-1.5 spells out
    # counts and ignores option letters) cannot be calibrated at generation time: excluded
    s = s[s.nodec <= 0.10]
    s["off_cal"] = s.off_cal.str.replace("±", r"$\pm$", regex=False)
    star = lambda p: "$^{*}$" if float(p) < 0.05 else ""

    def block(name, label, caption):
        x = s[s.set == name].sort_values("o")
        rows = [f"{MODEL[r.model]} & {r.raw:.1f} & {r.remap:.1f} & \\textbf{{{r.cal:.1f}}}{star(r.p_raw)} & {r.off_cal} \\\\"
                for r in x.itertuples()]
        n = int(x.n.iloc[0])
        return rf"""\begin{{table}}[t]
\centering\small
\caption{{{caption} Test half ($n={n}$); $^{{*}}$: calibrated vs.\ raw, exact McNemar $p<0.05$. Last column: offline calibrator, mean$\pm$sd over 20 stratified splits.}}
\label{{{label}}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr}}
\toprule
Model & Raw & \remap & Calib.\ (gen.) & Calib.\ (20 splits) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(f"{T}/gen_countbench.tex", "w").write(block("countbench", "tab:gen_cb",
        r"\textbf{Calibrated free-form generation on CountBench} (counts 1--9, prompt of~\cite{arxiv260709544})."))
    # VLMBias: each image appears at 3 resolutions x 2 wordings; random splits put copies on both halves (leak).
    g = pd.read_csv(f"{R}/vlmbias_grouped.csv")
    g = g[g.model.isin(s.model[s.set == "illusions"])].assign(o=lambda d: d.model.map(ORDER.index))
    gi = {k: g[(g.set == "illusions") & (g.split == k)].set_index("model") for k in ("ungrouped", "grouped")}
    rows = []
    for r in s[s.set == "illusions"].sort_values("o").itertuples():
        u, v = gi["ungrouped"].loc[r.model], gi["grouped"].loc[r.model]
        rows.append(f"{MODEL[r.model]} & {r.raw:.1f} & {r.cal:.1f}{star(r.p_raw)} & {u.raw:.1f}$\\to${u.cal:.1f} & "
                    f"{v.raw:.1f}$\\to${v.cal:.1f} & {v.gain:+.1f}$\\pm${v.gain_sd:.1f} \\\\")
    open(f"{T}/gen_illusions.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{VLMBias illusion subsets: a split leak.} Every image appears at three resolutions and with two question wordings, all with the same answer. With random splits (our first analysis; left) copies fall on both halves and calibrated generation appears to help; with splits by image (right) the gain disappears. Raw$\to$calibrated, offline, mean over 20 splits; $^{*}$: McNemar $p<0.05$ on the single generated split.}
\label{tab:gen_ill}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
& \multicolumn{3}{c}{Random split (leaky)} & \multicolumn{2}{c}{Split by image} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Model & Raw & Gen. & 20 splits & 20 splits & Gain \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    # MMVP: label-symmetric by construction (each pair has opposite answers); item and pair accuracy
    x = s[s.set == "MMVP"].sort_values("o")
    if len(x):
        rows = [f"{MODEL[r.model]} & {r.raw:.1f} & {r.cal:.1f}{star(r.p_raw)} & {r.off_raw.split('±')[0]} & {r.off_cal.split('$')[0]} & "
                f"{r.pair_raw:.1f} & {r.pair_cal:.1f} \\\\" for r in x.itertuples()]
        open(f"{T}/gen_mmvp.tex", "w").write(rf"""\begin{{table}}[t]
\centering\small
\caption{{\textbf{{Calibrated generation on MMVP}}~\cite{{tong2024eyes}}, where every image pair has opposite answers, so neither a prior nor \remap can help (\remap = raw for all models). Test half split by pair ($n={int(x.n.iloc[0])}$ questions; pair accuracy on it); $^{{*}}$: McNemar $p<0.05$ vs.\ raw. Gains are small: MMVP errors are mostly perceptual.}}
\label{{tab:gen_mmvp}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
& \multicolumn{{2}}{{c}}{{Test half}} & \multicolumn{{2}}{{c}}{{20 splits (mean)}} & \multicolumn{{2}}{{c}}{{Pair acc.}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}
Model & Raw & Calib. & Raw & Calib. & Raw & Calib. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    # POPE: balanced Yes/No (yes-ratio 50%); its known failure is a yes/no bias, i.e. a boundary shift
    x = s[s.set == "POPE"].sort_values("o")
    if len(x):
        rows = [f"{MODEL[r.model]} & {r.raw:.1f} & {r.remap:.1f} & {r.cal:.1f}{star(r.p_raw)} & {r.off_raw.split('±')[0]}$\\to${r.off_cal.split('$')[0]} & "
                f"{r.yes_raw:.0f}$\\to${r.yes_cal:.0f} & {r.adv_raw:.1f}$\\to${r.adv_cal:.1f} \\\\" for r in x.itertuples()]
        open(f"{T}/gen_pope.tex", "w").write(rf"""\begin{{table}}[t]
\centering\small
\caption{{\textbf{{Calibrated generation on POPE}}~\cite{{li2023pope}} (balanced Yes/No; 3$\times$3000 questions, split by image, test $n={int(x.n.iloc[0])}$). $^{{*}}$: McNemar $p<0.05$ vs.\ raw. Yes\,\%: share of ``Yes'' answers (gold: 50). Adv.: the adversarial setting.}}
\label{{tab:gen_pope}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
Model & Raw & \remap & Calib. & 20 splits & Yes\,\% & Adv. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    # asymmetric subsets: remap vs calibrated, offline, 20 splits by image (models/cells as in the generation tables)
    x = g[(g.split == "grouped") & g.set.isin(["Car Logos", "Dice Pattern", "Tally Pattern"])]
    x = x[[((m, k) in set(zip(s.model, s.set))) for m, k in zip(x.model, x.set)]]
    piv = {k: x[x.set == k].set_index("model") for k in ["Car Logos", "Dice Pattern", "Tally Pattern"]}
    rows = []
    for m in sorted(set().union(*[v.index for v in piv.values()]), key=ORDER.index):
        cells = []
        for k in piv:
            if m not in piv[k].index:  # excluded: >10% of items without a decision step
                cells.append("-- & -- & --")
                continue
            r = piv[k].loc[m]
            cells.append(f"{r.raw:.1f} & {r.remap:.1f} & {r.cal:.1f}")
        rows.append(f"{MODEL[m]} & " + " & ".join(cells) + r" \\")
    open(f"{T}/gen_asym.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{VLMBias subsets whose answer values are not symmetric} (some values are never correct). Raw / \remap / calibrated, offline on the decision-step scores, mean over 20 splits by image. \remap never looks at the image, yet reaches most of what a label-fitted fix achieves. --: $>$10\% of items never reach a decision step (format not followed); excluded.}
\label{tab:gen_asym}
\begin{tabular}{lrrrrrrrrr}
\toprule
& \multicolumn{3}{c}{Car Logos} & \multicolumn{3}{c}{Dice Pattern} & \multicolumn{3}{c}{Tally Pattern} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Model & Raw & \remap & Calib. & Raw & \remap & Calib. & Raw & \remap & Calib. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def g3():
    """Decomposition at the generation decision step (docs/g3_protocol.md): per-set summary (main) + per cell (supp)."""
    t = pd.read_csv(f"{R}/g3_table.csv").merge(pd.read_csv(f"{R}/g3_posthoc.csv")[["cell", "probe_last"]], on="cell", how="left")
    t = t.merge(pd.read_csv(f"{R}/g3_blank.csv")[["cell", "text_probe", "loss_txt"]].rename(columns={"loss_txt": "loss_img"}), on="cell", how="left")
    t["set"] = t.cell.str.rsplit("_", n=1).str[1]
    t["model_key"] = [c[: -len(s) - 1] for c, s in zip(t.cell, t.set)]
    ok = t[(t.n_err >= 30) & (t.nodec <= 0.10) & (t.text < 0.75)].copy()
    NAME = {"countbench": "CountBench", "illusions": "VLMBias illusions", "tally": "VLMBias Tally", "pope": "POPE", "mmvp": "MMVP"}
    rows = []
    for k in NAME:
        g = ok[ok.set == k]
        rows.append(f"{NAME[k]} & {len(g)} & {int((g.cal_recovers >= .5).sum())} & {100 * g.model.min():.0f}--{100 * g.model.max():.0f} & "
                    f"{100 * g.probe_last.min():.0f}--{100 * g.probe_last.max():.0f} & {100 * g.bias.median():.0f} & "
                    f"{100 * g.loss.median():.0f} ({100 * g.loss_img.median():.0f}) & {100 * g.perc.median():.0f} & {100 * g.text_probe.median():.0f} \\\\")
    open(f"{T}/g3.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{The diagnostic at the decision step of free-form generation} (pre-registered; 9 models, """ + str(len(ok)) + r""" cells with $\geq$30 errors; cross-fitting grouped by image). Rec.: cells where calibration recovers $\geq$50\% of the probe gap. Probe: best layer at the answer position. \bias/\loss/\perc: median \% of errors; in brackets, \loss on errors that a text-only probe (same prompt, blank image) gets wrong. Text: that probe's accuracy (median). Cells whose answer is predictable from the exact prompt ($\geq$75\%, \eg Car Logos) are excluded.}
\label{tab:g3}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Set & Cells & Rec. & Model & Probe & \bias & \loss & \perc & Text \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    ok["o"] = ok.model_key.map(ORDER.index)
    ok["t"] = ok.set.map(list(NAME).index)
    rows, last = [], None
    for r in ok.sort_values(["t", "o"]).itertuples():
        if r.set != last and last is not None:
            rows.append(r"\midrule")
        name = NAME[r.set] if r.set != last else ""
        last = r.set
        rec = "--" if pd.isna(r.cal_recovers) else "$>$100" if r.cal_recovers > 1 else "$<$0" if r.cal_recovers < 0 else f"{100 * r.cal_recovers:.0f}"
        rows.append(f"{name} & {MODEL[r.model_key]} & {r.n_err} & {100 * r.model:.1f} & {100 * r.calib:.1f} & {100 * r.probe:.1f} & {rec} & "
                    f"{100 * r.bias:.0f} & {100 * r.loss:.0f} & {100 * r.loss_img:.0f} & {100 * r.perc:.0f} \\\\")
    open(f"{T}/supp_g3.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Per-cell decomposition at the generation decision step.} As \cref{tab:decomp}, with 5-fold cross-fitting grouped by image. \loss$_\text{img}$: \loss errors on which the text-only probe (blank image) is wrong.}
\label{tab:supp_g3}
\begin{tabular}{llrrrrrrrrr}
\toprule
Set & Model & \#err & Model & Calib & Probe & Rec.\,(\%) & \bias & \loss & \loss$_\text{img}$ & \perc \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    print("g3:", int((ok.cal_recovers >= .5).sum()), "/", len(ok))


def steer():
    """Probe-guided steering at the decision step vs calibration (G3 addendum 3; alpha=16 post hoc)."""
    d = pd.read_csv(f"{R}/g3_steer.csv")
    t = pd.read_csv(f"{R}/g3_table.csv").set_index("cell")
    d = d[d.cell.map(lambda c: t.nodec[c] <= 0.10)]
    d["set"] = d.cell.str.rsplit("_", n=1).str[1]
    rows = []
    for k, name in (("illusions", r"Illusions (\loss)"), ("countbench", r"CountBench (\bias)")):
        g = d[d.set == k]
        f = lambda c: f"{100 * g[c].median():.1f}"
        rows.append(f"{name} & {len(g)} & {f('raw')} & {f('calib')} & {f('steer_a1')} & {f('steer_a16')} & "
                    f"{f('fixed_a16')} & {f('broken_a16')} \\\\")
    open(f"{T}/steer.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{Each error type is repaired by its own tool.} Medians over models. Steering adds $\alpha(\mu_p-\mu_m)$ (class means of the answer-position state at 75\% depth, cross-fitted by image) when the probe's class $p$ differs from the answer $m$. $\alpha{=}1$ was pre-registered; $\alpha{=}16$ was chosen after seeing $\alpha{=}1$. Fixed / broken: \% of errors / of correct answers changed by $\alpha{=}16$.}
\label{tab:steer}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
& & & & \multicolumn{2}{c}{Steering} & & \\
\cmidrule(lr){5-6}
Set & Cells & Raw & Calib. & $\alpha{=}1$ & $\alpha{=}16$ & Fixed & Broken \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    # per cell (supp)
    d["model_key"] = [c[: -len(s) - 1] for c, s in zip(d.cell, d.set)]
    d["o"], d["t"] = d.model_key.map(ORDER.index), d.set.map(["illusions", "countbench"].index)
    rows, last = [], None
    for r in d.sort_values(["t", "o"]).itertuples():
        if r.set != last and last is not None:
            rows.append(r"\midrule")
        name = {"illusions": "Illusions", "countbench": "CountBench"}[r.set] if r.set != last else ""
        last = r.set
        rows.append(f"{name} & {MODEL[r.model_key]} & {100 * r.raw:.1f} & {100 * r.calib:.1f} & {100 * r.probe_L:.1f} & "
                    f"{100 * r.steer_a1:.1f} & {100 * r.broken_a1:.1f} & {100 * r.steer_a16:.1f} & {100 * r.fixed_a16:.1f} & {100 * r.broken_a16:.1f} \\\\")
    open(f"{T}/supp_steer.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Steering at the decision step, per cell.} Accuracy (\%) of the raw model, the calibrator and the probe at the steered layer (75\% depth), all cross-fitted by image; steered accuracy at $\alpha{=}1$ (pre-registered) and $\alpha{=}16$ (post hoc), with the \% of errors fixed and of correct answers broken.}
\label{tab:supp_steer}
\begin{tabular}{llrrrrrrrr}
\toprule
& & & & & \multicolumn{2}{c}{$\alpha{=}1$} & \multicolumn{3}{c}{$\alpha{=}16$} \\
\cmidrule(lr){6-7}\cmidrule(lr){8-10}
Set & Model & Raw & Calib & Probe & Acc & Broken & Acc & Fixed & Broken \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def valid():
    """R1 (docs/r1_protocol.md, docs/r1_report.md) + G3 addendum 4: tests of the diagnostic itself."""
    L = pd.read_csv(f"{R}/r1_ladder.csv")
    cf, sc = pd.read_csv(f"{R}/r1_cf.csv"), pd.read_csv(f"{R}/r1_steer_ctrl.csv")
    wu = pd.read_csv(f"{R}/r2_whatsup.csv"); wu = wu[wu.n_loss >= 20]  # R2 E: What'sUp groups
    al = pd.read_csv(f"{R}/r2_alpha.csv")  # R2 F: alpha per model on validation images
    al["f1"] = al.il_sel >= np.maximum(al.il_raw, al.il_cal) + 0.10
    ho = pd.read_csv(f"{R}/g3_heldout.csv")
    ho = ho[(ho.n_err >= 30) & (ho.nodec <= 0.10) & (ho.text < 0.75)]
    hi, hc = ho[ho.cell.str.endswith("_illusions")], ho[ho.cell.str.endswith("_countbench")]
    frac = lambda hit: f"{int(hit.sum())}/{len(hit)}"
    ok = lambda b: r"\cmark" if b else r"\xmark"

    def per(t):
        gain = t.acc_C - t.model
        a = t[gain >= 0.02]
        return dict(a1=(a.acc_D - a.model) >= 0.8 * (a.acc_C - a.model), a2=(t.acc_M - t.acc_C) <= 0.02,
                    a3=(t.acc_PM - t.acc_P) <= 0.02, nb=int((t.rec_best >= .5).sum()), nn=int((t.rec_nested >= .5).sum()),
                    atF=t.loss_atF.median())
    c, g = per(L[~L.g3]), per(L[L.g3])
    b = cf[cf.n_loss >= 20]
    gq = cf[(cf.para_in - cf.para_raw) >= 0.02]
    c1 = (gq.para_transfer - gq.para_raw) >= 0.5 * (gq.para_in - gq.para_raw)
    il = sc[sc.set == "illusions"]
    d1 = ((il.random - il.raw).abs() <= .05) & ((il.shuffle - il.raw).abs() <= .05)
    h1 = hi.steer_a16 >= np.maximum(hi.model, hi.calib) + 0.10
    h3 = pd.concat([hi.loss > hi.bias, hc.bias > hc.loss])
    rows = [
        (r"\multicolumn{5}{l}{\emph{Readout capacity}}", None),
        (r"MLP on logits adds $\leq$2 pts", frac(c["a2"]), frac(g["a2"]), ok(c["a2"].mean() >= 2/3 and g["a2"].mean() >= 2/3)),
        (r"MLP probe adds $\leq$2 pts", frac(c["a3"]), frac(g["a3"]), ok(c["a3"].mean() >= 2/3 and g["a3"].mean() >= 2/3)),
        (r"Offset alone: $\geq$80\% of cal.\ gain", frac(c["a1"]), frac(g["a1"]), ok(False)),
        (r"Nested layer: cells Rec.\ $\geq$50\%", f"{c['nb']}$\\to${c['nn']}", f"{g['nb']}$\\to${g['nn']}", ok(False) + r"$^\dagger$"),
        (r"\loss decodable at last layer", f"{100 * c['atF']:.0f}\\%", f"{100 * g['atF']:.0f}\\%", "--"),
        (r"\multicolumn{5}{l}{\emph{Counterfactual images}}", None),
        (r"Synth.: probe right on partner $\geq$70\%", f"{100 * b.loss_probe_ok.median():.0f}\\%", "--", ok(b.loss_probe_ok.median() >= .7)),
        (r"Synth.: model right on it less often", frac(b.loss_model_ok < b.loss_probe_ok), "--", ok((b.loss_model_ok < b.loss_probe_ok).mean() >= 2/3)),
        (r"Real$^\ddagger$: probe right on partner $\geq$70\%", f"{100 * wu.loss_probe_ok.median():.0f}\\%", "--", ok(wu.loss_probe_ok.median() >= .7)),
        (r"Real$^\ddagger$: model right on it less often", frac(wu.loss_model_ok < wu.loss_probe_ok), "--", ok((wu.loss_model_ok < wu.loss_probe_ok).mean() >= 2/3)),
        (r"\multicolumn{5}{l}{\emph{Calibrator transfer}}", None),
        (r"Paraphrase keeps $\geq$50\% of gain", frac(c1), "--", ok(c1.mean() >= 2/3)),
        (r"\multicolumn{5}{l}{\emph{Steering ($\alpha{=}16$)}}", None),
        (r"Random/shuffled dir.: illus.\ $\pm$5 pts", "--", frac(d1), ok(d1.all())),
        (r"Held-out: illus.\ $\geq$ raw, cal.\ $+10$", "--", frac(h1), ok(h1.mean() >= 2/3)),
        (r"Held-out: CountBench $<$ cal.", "--", frac(hc.steer_a16 < hc.calib), ok((hc.steer_a16 < hc.calib).mean() >= 2/3)),
        (r"Held-out: decomposition replicates", "--", frac(h3), ok(h3.mean() >= 2/3)),
        (r"\multicolumn{5}{l}{\emph{Steering, $\alpha$ chosen on validation images}}", None),
        (r"Held-out: illus.\ $\geq$ raw, cal.\ $+10$ (3/4)", "--", frac(al[al.set == "heldout"].f1), ok(al[al.set == "heldout"].f1.sum() >= 3)),
        (r"Finished models: same bar", "--", frac(al[al.ill_valid].f1), ok(al[al.ill_valid].f1.mean() >= 2/3)),
        (r"CountBench $<$ cal.\ at that $\alpha$", "--", frac(al.cb_sel < al.cb_cal), ok((al.cb_sel < al.cb_cal).mean() >= 2/3)),
    ]
    body = []
    for r in rows:
        body.append(r[0] + r" \\" if r[1] is None else " & ".join(r) + r" \\")
    open(f"{T}/valid.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{Testing the diagnostic.} Each line was pre-registered with its bar before it was run (supplementary); counts are cells meeting the bar, \cmark/\xmark\ whether $\geq$2/3 did. Closed-form: 24 cells (counterfactual images: synthetic tasks, 4 cells with $\geq$20 \loss errors); Gen.: 40 decision-step cells; held-out: Qwen3.5-2B/4B/9B and Qwen2-VL-2B, never used before, $\alpha$ fixed at 16. $^\dagger$Fails in the direction that favours calibration. $^\ddagger$What'sUp: 4 photos of one object pair in 4 relations, same prompt; folds by object pair; 4 cells with $\geq$20 \loss errors (13 models), added after the synthetic test. $\alpha$ on validation: 7 of 13 models finished (3 of 4 held-out); F1 fails whatever the rest give.}
\label{tab:valid}
\footnotesize
\begin{tabular}{@{}lrrc@{}}
\toprule
Test (bar) & Closed & Gen. & \\
\midrule
""".replace(r"\multicolumn{5}", r"\multicolumn{4}") + "\n".join(body).replace(r"\multicolumn{5}", r"\multicolumn{4}") + r"""
\bottomrule
\end{tabular}
\end{table}
""")

def supp_r1():
    """R1 supplementary tables: held-out steering (G3 addendum 4) and steering controls."""
    NAME = {"qwen35_2b": "Qwen3.5-2B", "qwen35_4b": "Qwen3.5-4B", "qwen35_9b": "Qwen3.5-9B", "qwen2vl_2b": "Qwen2-VL-2B"}
    ho = pd.read_csv(f"{R}/g3_heldout.csv")
    ho["set"] = ho.cell.str.rsplit("_", n=1).str[1]
    ho["m"] = [c[: -len(s) - 1] for c, s in zip(ho.cell, ho.set)]
    rows = []
    for m in NAME:
        r = [NAME[m]]
        for s in ("illusions", "countbench"):
            x = ho[(ho.m == m) & (ho.set == s)].iloc[0]
            r += [f"{100 * x[k]:.1f}" for k in ("model", "calib", "probe", "steer_a1", "steer_a16")] + [f"{100 * x.loss:.0f}/{100 * x.bias:.0f}"]
        rows.append(" & ".join(r) + r" \\")
    sc = pd.read_csv(f"{R}/r1_steer_ctrl.csv")
    rows2 = [" & ".join([MODEL[r.model], {"illusions": "Illusions", "countbench": "CountBench"}[r.set]] +
                        [f"{100 * getattr(r, k):.1f}" for k in ("raw", "real", "random", "shuffle")]) + r" \\" for r in sc.itertuples()]
    open(f"{T}/supp_r1.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Held-out models} (never used before; $\alpha$ fixed at 16 in advance). Accuracy (\%) of the raw model, the calibrator and the best-layer probe, and after steering at $\alpha{=}1$ and $\alpha{=}16$; \loss/\bias: \% of errors.}
\label{tab:supp_heldout}
\begin{tabular}{l|rrrrrr|rrrrrr}
\toprule
& \multicolumn{6}{c|}{Illusions} & \multicolumn{6}{c}{CountBench} \\
Model & Raw & Cal. & Probe & $\alpha{=}1$ & $\alpha{=}16$ & \loss/\bias & Raw & Cal. & Probe & $\alpha{=}1$ & $\alpha{=}16$ & \loss/\bias \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\vspace{6pt}

\caption{\textbf{Steering controls} at $\alpha{=}16$: the probe's direction, a random direction of the same norm, and class means fitted on shuffled labels (same gating).}
\label{tab:supp_ctrl}
\begin{tabular}{llrrrr}
\toprule
Model & Set & Raw & Probe dir. & Random & Shuffled \\
\midrule
""" + "\n".join(rows2) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

def supp_alpha():
    """R2 F: steering strength chosen per model on validation images (docs/r2_protocol.md)."""
    NAME = dict(MODEL, qwen35_2b="Qwen3.5-2B", qwen35_4b="Qwen3.5-4B", qwen35_9b="Qwen3.5-9B", qwen2vl_2b="Qwen2-VL-2B")
    al = pd.read_csv(f"{R}/r2_alpha.csv")
    al["o"] = al.model.map(list(NAME).index)
    rows, last = [], None
    for r in al.sort_values(["set", "o"]).itertuples():
        if last is not None and r.set != last:
            rows.append(r"\midrule")
        last = r.set
        name = NAME[r.model] + ("" if r.ill_valid else r"$^*$")
        rows.append(" & ".join([name, f"{r.alpha_sel:g}"] + [f"{100 * getattr(r, k):.1f}" for k in ("il_raw", "il_cal", "il_a16", "il_sel", "cb_raw", "cb_cal", "cb_sel")]) + r" \\")
    open(f"{T}/supp_alpha.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{Steering strength chosen on validation images.} $\alpha\in\{0,1,\dots,64\}$ chosen per model on half of the illusion images, accuracy (\%) on the other half, mean of 20 splits by image; CountBench steered with the same $\alpha$. Top: development models (where $\alpha{=}16$ was chosen); bottom: held-out models. $^*$Cell excluded by the validity filter.}
\label{tab:supp_alpha}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lr|rrrr|rrr}
\toprule
& & \multicolumn{4}{c|}{Illusions (test images)} & \multicolumn{3}{c}{CountBench} \\
Model & $\alpha^*$ & Raw & Cal. & $\alpha{=}16$ & $\alpha^*$ & Raw & Cal. & $\alpha^*$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def supp():
    """Supplementary tables: few-label calibration of generation; decision-step coverage per cell."""
    f = pd.read_csv(f"{R}/fewshot_gen.csv")
    f = f[f.model.isin(MODEL)]
    nd = pd.read_csv(f"{R}/gencal_stats.csv").set_index(["model", "set"]).nodec  # same exclusion rule as the main tables
    f = f[[nd.get((m, k), 0) <= 0.10 for m, k in zip(f.model, f.set)]]
    f["o"] = f.model.map(ORDER.index)
    f = f[f.set != "illusions"]  # their train/test halves share images (VLMBias split leak); see tab:gen_ill
    SET = {"countbench": "CountBench", "MMVP": "MMVP", "POPE": "POPE"}
    rows, last = [], None
    for r in f.assign(t=f.set.map(list(SET).index)).sort_values(["t", "o"]).itertuples():
        if r.set != last and last is not None:
            rows.append(r"\midrule")
        name = SET[r.set] if r.set != last else ""
        last = r.set
        cells = [str(getattr(r, k)).replace("±", r"$\pm$") for k in ("_4", "_5", "_6", "_7")]
        rows.append(f"{name} & {MODEL[r.model]} & {r.raw:.1f} & " + " & ".join(cells) + f" & {r._8:.1f} \\\\")
    open(f"{T}/supp_fewshot.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{How many labels does generation-time calibration need?} Offline calibrator on decision-step scores fitted on $K$ labelled items per class, evaluated on the test half; mean$\pm$sd over 50 draws. Last column: the full training half, as in the main paper.}
\label{tab:supp_fewshot}
\begin{tabular}{llrrrrrr}
\toprule
Set & Model & Raw & $K{=}1$ & $K{=}2$ & $K{=}5$ & $K{=}10$ & Full \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    s = pd.read_csv(f"{R}/gencal_stats.csv")
    s = s[s.model.isin(MODEL)]
    sets = ["countbench", "illusions", "Car Logos", "Dice Pattern", "Tally Pattern", "MMVP", "POPE"]
    sets = [k for k in sets if k in set(s.set)]
    piv = s.pivot_table(index="model", columns="set", values="nodec")
    rows = []
    for m in sorted(piv.index, key=ORDER.index):
        cells = []
        for k in sets:
            v = piv.loc[m].get(k)
            cells.append("" if pd.isna(v) else (rf"\textbf{{{100 * v:.0f}}}" if v > 0.10 else f"{100 * v:.0f}"))
        rows.append(f"{MODEL[m]} & " + " & ".join(cells) + r" \\")
    head = " & ".join({"countbench": "CountBench", "illusions": "Illusions", "Car Logos": "Car", "Dice Pattern": "Dice",
                       "Tally Pattern": "Tally"}.get(k, k) for k in sets)
    open(f"{T}/supp_coverage.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{Items without a decision step} (\%), \ie the model never produced an answer-like token at a gated position (format not followed); these keep their raw answer. Bold: $>$10\%, cell excluded from the main tables. Empty: not run.}
\label{tab:supp_coverage}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l""" + "r" * len(sets) + r"""}
\toprule
Model & """ + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


if __name__ == "__main__":
    os.makedirs(T, exist_ok=True)
    decomp()
    gencal()
    g3()
    steer()
    valid()
    supp_r1()
    supp_alpha()
    supp()
