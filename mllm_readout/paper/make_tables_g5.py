"""G5 tables and figure (docs/g5_protocol.md): shortcut-controlled illusions, interchange by depth, routing, W4.
Run from mllm_readout/: PYTHONPATH=paper python paper/make_tables_g5.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_tables import R, T
from make_tables_g4 import MODEL, ORDER, med

INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED})
p = lambda v: "--" if pd.isna(v) else f"{100 * v:.0f}"


def syn():
    import os
    cells = {}
    SETS = [s for s in ("illusyn", "illusynL", "illusynC") if os.path.exists(f"{R}/g5/syn_{s}.csv")]
    RIGHT = "illusynC" if "illusynC" in SETS else "illusynL"  # main table: G6 confirmatory set when available
    for s in SETS:
        d = pd.read_csv(f"{R}/g5/syn_{s}.csv")
        g, m = d.iloc[0], d.iloc[1:]
        mm = lambda c: f"{100 * m[c].median():.0f}" if c in m and m[c].notna().any() else "--"
        cells[s] = [(p(g.pixel_within), p(g.pixel_cross)), (p(g.siglip2_within), p(g.siglip2_cross)),
                    (p(g.dinov2_within), p(g.dinov2_cross)), (mm("pre_within"), mm("pre_cross")),
                    (mm("l0_within"), mm("l0_cross")), (mm("dec_within"), mm("dec_cross")), (mm("raw"), "--"),
                    (mm("l0_mlp_within"), mm("l0_mlp_cross")),
                    (p(g.siglip2_signed_within), p(g.siglip2_signed_cross)), (mm("l0_signed_within"), mm("l0_signed_cross")),
                    (mm("dec_signed_within"), mm("dec_signed_cross"))]
        cells[s + "_n"] = len(m)
    names = ["Pixel statistics (64)", "SigLIP2 (frozen)", "DINOv2 (frozen)", "Own vision encoder output", "Own LLM input",
             "Decision state, 75\\% depth", "Model output", "Own LLM input, MLP probe",
             "SigLIP2 (frozen)", "Own LLM input", "Decision state, 75\\% depth"]
    rows = []
    for i, n in enumerate(names):
        if i == 0:
            rows.append(r"\multicolumn{5}{@{}l}{\emph{Equal vs.\ unequal}} \\")
        if i == 8:
            rows.append(r"\multicolumn{5}{@{}l}{\emph{Which line is longer (unequal items)}} \\")
        a, b = cells["illusyn"][i], cells[RIGHT][i]
        rows.append(f"{n} & {a[0]} & {a[1]} & {b[0]} & {b[1]} \\\\")
    open(f"{T}/syn.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{Separable is not seen.} Procedural illusions (Müller-Lyer, Ponzo, vertical--horizontal) with the two compared lengths summing to a constant and the illusion context independent of which line is longer, in two nuisance regimes (A: black on white, thin; B: coloured, grey or noise background, thick, rotated). Accuracy (\%) within a regime (5-fold CV) and across regimes (train on one, test on the other); MLLM rows: median over """ + f"{cells['illusyn_n']} / {cells[RIGHT + '_n']}" + r""" models. Chance is 50. Left: relative difference $|\Delta|\in[0.10,0.30]$ (G5, pre-registered); right: the VLMBias range, $|\Delta|\in[0.30,0.50]$, """ + ("fresh stimuli, pre-registered after an exploratory run (G6)" if RIGHT == "illusynC" else "exploratory") + r""".}
\label{tab:syn}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
& \multicolumn{2}{c}{$|\Delta|$ 0.10--0.30} & \multicolumn{2}{c}{$|\Delta|$ 0.30--0.50} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Readout & within & across & within & across \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # supplementary: per model
    rows = []
    for s, lab in [(s, l) for s, l in (("illusyn", r"$|\Delta|$ 0.10--0.30 (G5, pre-registered)"), ("illusynL", r"$|\Delta|$ 0.30--0.50 (exploratory, seed 2027)"),
                                       ("illusynC", r"$|\Delta|$ 0.30--0.50 (G6, fresh stimuli, pre-registered)")) if s in SETS]:
        m = pd.read_csv(f"{R}/g5/syn_{s}.csv").iloc[1:]
        m = m.assign(o=m.model.map(ORDER.index)).sort_values("o")
        rows.append((r"\midrule" + "\n" if rows else "") + rf"\multicolumn{{12}}{{@{{}}l}}{{\emph{{{lab}}}}} \\")
        for r in m.itertuples():
            rows.append(f"{MODEL[r.model]} & {p(r.raw)} & {p(r.say_no)} & {p(r.l0_within)} & {p(r.l0_cross)} & {p(r.dec_within)} & "
                        f"{p(r.dec_cross)} & " + " & ".join(str(int(getattr(r, t))) for t in ("BIAS", "UNSPOKEN", "INHERITED", "DROPPED", "UNSEEN")) + r" \\")
    open(f"{T}/supp_syn.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Shortcut-controlled illusions per model} (G5 S). Raw: model accuracy; ``No'': share of ``No'' (unequal) answers; probe accuracies within / across nuisance regimes (\%); error types with the model's own LLM input (projector output) as $E$, calibrator and decision-state probe cross-fitted on the same items. With readouts near chance (left block), the types are not interpretable; S4 is reported as registered.}
\label{tab:supp_syn}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrrrr@{}}
\toprule
Model & Raw & ``No'' & L0 within & L0 across & Dec.\ within & Dec.\ across & \textsc{bias} & \textsc{unsp.} & \textsc{inh.} & \textsc{drop.} & \textsc{unseen} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


FR = [0.125, 0.25, 0.5, 0.75]


def swap():
    d = pd.read_csv(f"{R}/g5/swap.csv")
    get = lambda c: d[c] if c in d else pd.Series([np.nan] * len(d))
    probe = np.stack([d["M1_probe_follows@0"]] + [get(f"probe->partner swap@{100 * f:g}") for f in FR], 1)
    ans = np.stack([d["M2_answer_follows@0"]] + [get(f"answer->partner swap@{100 * f:g}") for f in FR], 1)
    ctrl = np.stack([d["M3_ctrl_probe_changes@0"], get("probe changes ctrl@25"), get("probe changes ctrl@75")], 1)
    x = np.array([0] + [100 * f for f in FR])
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for i in range(len(d)):
        ok = ~np.isnan(probe[i])
        ax.plot(x[ok], 100 * probe[i][ok], color=BLUE, lw=0.5, alpha=0.35)
        ax.plot(x[ok], 100 * ans[i][ok], color=ORANGE, lw=0.5, alpha=0.35)
    ax.plot(x, 100 * np.nanmedian(probe, 0), "-o", color=BLUE, ms=3, lw=1.5, label="decision-state probe follows the partner")
    ax.plot(x, 100 * np.nanmedian(ans, 0), "-o", color=ORANGE, ms=3, lw=1.5, label="model's answer follows the partner")
    ax.plot([0, 25, 75], 100 * np.nanmedian(ctrl, 0), "s", color=GREEN, ms=3.5, mfc="white", mew=1,
            label="same-label control: probe changes")
    nofx = 100 * np.median(1 - d.probe_base)
    ax.axhline(nofx, color=MUTED, lw=0.6, ls=":")
    ax.text(76, nofx + 1.5, "no effect", fontsize=5.5, color=MUTED, ha="right")
    ax.set_xticks(x, ["0\n(partner run)", "12.5", "25", "50", "75"])
    ax.set_xlabel("depth of the image-position swap (% of layers)", color=INK)
    ax.set_ylabel("% of pairs", color=INK)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=False, fontsize=5.3, labelspacing=0.2, borderaxespad=0, handlelength=1.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig("paper/fig/swap_g5.pdf", bbox_inches="tight")
    fig.savefig("paper/fig/swap_g5.png", dpi=200, bbox_inches="tight")
    print("swap medians: probe", np.round(100 * np.nanmedian(probe, 0)), "answer", np.round(100 * np.nanmedian(ans, 0)),
          "ctrl", np.round(100 * np.nanmedian(ctrl, 0)), "no effect", round(nofx), "n", len(d))
    cols = [("M1_probe_follows@0", "P@0"), ("M2_answer_follows@0", "A@0")] + \
           [(f"probe->partner swap@{100 * f:g}", f"P@{100 * f:g}") for f in FR] + \
           [(f"answer->partner swap@{100 * f:g}", f"A@{100 * f:g}") for f in FR] + \
           [("M3_ctrl_probe_changes@0", "C@0"), ("probe changes ctrl@25", "C@25"), ("probe changes ctrl@75", "C@75")]
    D = pd.read_csv(f"{R}/g5/swap.csv")
    D = D.assign(o=D.model.map(ORDER.index)).sort_values("o")
    rows = [f"{MODEL[r['model']]} & {int(r['n_pairs'])} & " + " & ".join(p(r.get(c, np.nan)) for c, _ in cols) + r" \\"
            for _, r in D.iterrows()]
    open(f"{T}/supp_swap.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Interchange by depth, per model} (G5 M; \%). Image-position states of an illusion item are replaced, at the input of the layer at the given depth, by those of a partner with the other label (same type, strength, wording and token grid). P: the decision-state probe (75\% depth, trained on unedited states, folds grouped by type and strength) predicts the partner's label; A: the model's answer is the partner's label; C: same-label control, the probe's prediction changes. Depth 0 is the pre-registered swap, identical to running the partner item.}
\label{tab:supp_swap}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lr""" + "r" * len(cols) + r"""@{}}
\toprule
Model & Pairs & """ + " & ".join(n for _, n in cols) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def route():
    d = pd.read_csv(f"{R}/g5/route.csv")
    d = d.assign(o=d.model.map(ORDER.index), s=d.set.map({"illusions": 0, "countbench": 1, "mmvp": 2})).sort_values(["s", "o"])
    SN = {"illusions": "Illusions", "countbench": "CountBench", "mmvp": "MMVP"}
    rows = [f"{SN[r.set]} & {MODEL[r.model]} & {p(r.raw)} & {p(r.best_single)} ({r.best}) & {p(r.rule)} & {p(r.learned)} & "
            f"{p(r.random)} & {p(r.oracle)} \\\\" for r in d.itertuples()]
    m = d[["raw", "best_single", "rule", "learned", "random", "oracle"]].mean()
    rows.append(r"\midrule" + "\n" + "Mean & & " + " & ".join(p(m[c]) for c in ["raw", "best_single", "rule", "learned", "random", "oracle"]) + r" \\")
    open(f"{T}/supp_route.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{End-to-end routed repair on held-out base images} (G5 R; accuracy \%). The router sees only whether the cross-fitted calibrator, decision-state probe and SigLIP2 probe agree with the model's answer. Best single: the best repair applied to all items, chosen post hoc (favours the baseline). Rule: calibrator disagrees $\to$ R1, else SigLIP2 disagrees $\to$ R3x, else keep. Learned: best repair per agreement pattern on half of the base images, tested on the other half (20 splits). Random: Learned's repair frequencies, assigned at random.}
\label{tab:supp_route}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Set & Model & Raw & Best single & Rule & Learned & Random & Oracle \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    print("route means", (100 * m).round(1).to_dict())


def w4():
    d = pd.read_csv(f"{R}/g5/w4.csv")
    g = d.groupby(["set", "target"])
    TN = {"own": "own probe", "ext": "SigLIP2 probe", "oracle": "true class", "anti": "wrong class", "random": "random dir.",
          "shuffle": "shuffled labels"}
    rows = []
    for s in ("illusions", "countbench"):
        for t in ("own", "ext", "oracle", "anti", "random", "shuffle"):
            if (s, t) not in g.groups:
                continue
            x = g.get_group((s, t))
            rows.append(f"{'Illusions' if s == 'illusions' else 'CountBench'} & {TN[t]} & {len(x)} & {med(x.raw)} & {med(x.steer)} & "
                        f"{med(x.target_acc)} & {med(100 * x.steer_minus_target, False)} & {med(x.written)} \\\\")
    open(f"{T}/supp_w4.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Steering versus outputting the target directly} (W4; median [min--max] over models, \%). Target: accuracy of replacing the model's answer by the steering target's class (no steering); Steer$-$target in points; Written: share of edited items whose answer becomes the target.}
\label{tab:supp_w4}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Set & Target & Models & Raw & Steering & Target & Steer$-$target & Written \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


if __name__ == "__main__":
    syn()
    swap()
    route()
    w4()
