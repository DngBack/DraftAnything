"""G7 tables and figure (docs/g7_protocol.md): geometric readout with a source-fixed rule, text-only comparison,
psychometric curves with an unseen regime, steering along the geometric direction.
Run from mllm_readout/: PYTHONPATH=paper python paper/make_tables_g7.py"""
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_tables import R, T
from make_tables_g4 import MODEL, ORDER
from make_tables_g5 import INK, MUTED, BLUE, ORANGE, GREEN, p

G5SET = ["qwen3vl4b", "qwen3vl8b", "qwen35_4b", "qwen35_9b", "gemma3_4b", "internvl35_8b", "llava15_7b", "qwen2vl_7b"]
READ = [("pixel", "Pixel statistics (64)"), ("siglip2", "SigLIP2 (frozen)"), ("dinov2", "DINOv2 (frozen)"),
        ("pre", "Own vision encoder output"), ("l0", "Own LLM input"), ("dec", "Decision state, 75\\% depth")]
srt = lambda ms: sorted(ms, key=lambda m: ORDER.index(m) if m in ORDER else 99)
OUTN = {}


def rng_(v):
    return f"{100 * v.min():.0f}--{100 * v.max():.0f}"


def geom():
    G = pd.read_csv(f"{R}/g7/geom_illusynC.csv")
    rows = []
    for k, name in READ:
        c = G[G.readout == k]
        if c.empty:
            continue
        f = (lambda col: p(c[col].iloc[0])) if c.model.iloc[0] == "(model-free)" else (lambda col: f"{100 * c[col].median():.0f}")
        corr = f"{c['corr'].median():.2f}"
        rows.append(f"{name} & {f('binary')} & {f('rule')} & {f('oracle')} & {corr} & {f('signed')} \\\\")
        OUTN[k] = dict(binary=rng_(c.binary), rule=rng_(c.rule), corr=f"{c['corr'].min():.2f}--{c['corr'].max():.2f}", signed=rng_(c.signed),
                       oracle=rng_(c.oracle), rule_med=f"{100 * c.rule.median():.0f}", binary_med=f"{100 * c.binary.median():.0f}")
    open(f"{T}/geom.tex", "w").write(r"""\begin{table}[t]
\centering\footnotesize
\caption{\textbf{A geometric readout does not rescue equality} (G7 G, synthetic illusions, $|\Delta|$ 0.30--0.50). Ridge regression of $\log(L_1/L_2)$, ``equal'' iff $|\hat r|\leq\tau$ with $\tau$ fixed on the source regime, accuracy on the other regime (mean of A$\to$B and B$\to$A), against the direct equal/unequal classifier. Oracle: $\tau$ chosen on the target. $r$: correlation of $\hat r$ with the true log-ratio on the target; signed: which line is longer. Median over 8 models where model-specific.}
\label{tab:geom}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
& \multicolumn{3}{c}{Equal vs.\ unequal, across} & \\
\cmidrule(lr){2-4}
Readout & classifier & rule & oracle & $r$ & signed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    rows = []
    M = G[G.model.isin(G5SET)]
    for m in srt(M.model.unique()):
        c = M[M.model == m].set_index("readout")
        rows.append(f"{MODEL.get(m, m)} & {p(c.loc['l0', 'model_acc'])} & " + " & ".join(
            f"{p(c.loc[k, 'binary'])} & {p(c.loc[k, 'rule'])} & {c.loc[k, 'corr']:.2f}" for k in ("pre", "l0", "dec")) + r" \\")
    for s in ("illusynL", "illusyn"):  # secondary sets, medians
        f = f"{R}/g7/geom_{s}.csv"
        if os.path.exists(f):
            S = pd.read_csv(f)
            OUTN[s] = {k: (rng_(S[S.readout == k].rule), rng_(S[S.readout == k].binary)) for k in S.readout.unique()}
    open(f"{T}/supp_geom.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Geometric readout per model} (G7 G, illusynC). Accuracy (\%) across nuisance regimes of the direct equal/unequal classifier and of the fixed-threshold rule on the ridge-regressed log length ratio; $r$: correlation of the regressed with the true log-ratio on the target regime. Model: the model's own accuracy.}
\label{tab:supp_geom}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
& & \multicolumn{3}{c}{Own vision encoder output} & \multicolumn{3}{c}{Own LLM input} & \multicolumn{3}{c}{Decision state} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}
Model & model & classifier & rule & $r$ & classifier & rule & $r$ & classifier & rule & $r$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def psy():
    P = pd.read_csv(f"{R}/g7/psy.csv").set_index("model")
    C = pd.read_csv(f"{R}/g7/psy_curves.csv")
    C = C[C.model.isin(G5SET)]
    fig, ax = plt.subplots(figsize=(3.3, 1.95))
    for g, col in (("A", BLUE), ("B", ORANGE), ("C", GREEN)):
        c = C[(C.source == "image") & (C.regime == g)].groupby("diff").share_no.median()
        ax.plot(c.index, 100 * c.values, color=col, lw=1.6, marker="o", ms=2.5, label=f"model, regime {g}")
    c = C[(C.source == "text") & (C.regime == "all")].groupby("diff").share_no.median()
    if len(c):
        ax.plot(c.index, 100 * c.values, color=INK, lw=1.4, ls="--", label="model, lengths as text")
    c = C[(C.source == "l0 rule (CV)") & (C.regime == "all")].groupby("diff").share_no.median()
    if len(c):
        ax.plot(c.index, 100 * c.values, color=MUTED, lw=1.2, ls=":", label="own LLM input, geometric rule")
    ax.set_xlabel(r"relative length difference $|\Delta|$", color=INK)
    ax.set_ylabel('answers "unequal" (%)', color=INK)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout(pad=0.2)
    fig.savefig("paper/fig/psy.pdf"), fig.savefig("paper/fig/psy.png", dpi=200)
    Y3 = pd.read_csv(f"{R}/g7/geom_illusynP.csv").set_index(["model", "readout"]) if os.path.exists(f"{R}/g7/geom_illusynP.csv") else None
    rows = []
    for m in srt([m for m in P.index if m in G5SET]):
        r = P.loc[m]
        y3 = f"{p(Y3.loc[(m, 'l0'), 'rule'])} & {p(Y3.loc[(m, 'l0'), 'binary'])}" if Y3 is not None and (m, "l0") in Y3.index else "-- & --"
        d75 = "--" if not np.isfinite(r.img_d75) else f"{r.img_d75:g}"
        rows.append(f"{MODEL.get(m, m)} & {p(r.img_acc)} & {p(r.img_fa_A)} & {p(r.img_fa_B)} & {p(r.img_fa_C)} & {d75} & {r.Y1_p:.0e} & {r.Y2_p:.0e} & "
                    f"{p(r.get('txt_acc_eq'))} & {p(r.get('txt_acc_ge10'))} & {p(r.get('txt_acc'))} & {y3} \\\\")
    open(f"{T}/supp_psy.tex", "w").write(r"""\begin{table*}[t]
\centering\scriptsize
\caption{\textbf{Psychometric and text-only tests per model} (G7 Y, T; illusynP, $|\Delta|\in\{0,0.05,\dots,0.80\}$, regimes A, B and the unseen C). Image: accuracy, false-alarm rate (``unequal'' on equal items) per regime, the smallest $|\Delta|$ answered ``unequal'' $\geq$75\% (--: none), likelihood-ratio $p$ of the slope on $|\Delta|$ (Y1) and of regime indicators (Y2). Text: accuracy with the two lengths given as integers, on equal items, on $|\Delta|\geq0.10$, and overall. Y3: on regime C, the own-LLM-input geometric rule and the binary classifier, both fitted on A$\cup$B (\%).}
\label{tab:supp_psy}
\begin{tabular}{@{}lrrrrrrrrrrrr@{}}
\toprule
& \multicolumn{7}{c}{Image (G3 decision step)} & \multicolumn{3}{c}{Text only} & \multicolumn{2}{c}{Y3 (A$\cup$B$\to$C)} \\
\cmidrule(lr){2-8}\cmidrule(lr){9-11}\cmidrule(l){12-13}
Model & acc & FA$_A$ & FA$_B$ & FA$_C$ & $|\Delta|_{75}$ & $p$(Y1) & $p$(Y2) & equal & $\geq$.10 & all & rule & classifier \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    print("psy medians", P[[c for c in P.columns if P[c].dtype != object]].median().round(3).to_dict())


def steer():
    rows, res = [], {}
    for f in sorted(glob.glob(f"{R}/g7/steer_*.npz")):
        m = os.path.basename(f)[6:-4]
        if m.endswith("_smoke"):
            continue
        d = np.load(f)
        A, conds, al = d["ans"], d["conds"], d["alpha"]
        sh = [np.mean(A[:, k][A[:, k] >= 0] == 1) for k in range(len(conds))]
        res[m] = dict(zip([f"{c}@{a:g}" for c, a in zip(conds, al)], sh))
        rows.append(f"{MODEL.get(m, m)} & " + " & ".join(p(v) for v in sh) + f" & {float(d['cos_pm']):.2f} & {float(d['norm_dp']) / float(d['img_pos_norm']):.2f} \\\\")
    if not rows:
        return
    head = " & ".join(["none"] + [f"$d^{s}$ {a:g}" for s in "+-" for a in (0.5, 1, 2)] + ["random 2"])
    open(f"{T}/supp_steer.tex", "w").write(r"""\begin{table}[t]
\centering\scriptsize
\caption{\textbf{Steering along the geometric direction} (G7 I, exploratory; 240 equal illusynC items). Share of ``unequal'' answers (\%) after adding $\alpha d$ at every image position at the input of the layer at 25\% depth: $d^\pm$ the mean difference between items with the reference line $\geq$28\% longer (shorter) and equal items; random: a random direction of the norm of $d^+$. cos: cosine of $d^+$ and $d^-$; $\|d^+\|$ relative to the mean image-position state norm.}
\label{tab:supp_steer}
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
Model & """ + head + r""" & cos & $\|d^+\|$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    print("steer", {m: {k: round(v, 3) for k, v in r.items()} for m, r in res.items()})


if __name__ == "__main__":
    if os.path.exists(f"{R}/g7/geom_illusynC.csv"):
        geom()
    if os.path.exists(f"{R}/g7/psy.csv"):
        psy()
    steer()
    print({k: v for k, v in OUTN.items()})
