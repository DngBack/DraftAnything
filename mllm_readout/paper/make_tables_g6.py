"""G6 tables and figure (docs/g6_protocol.md): label-specific interchange by depth, price of repair.
Run from mllm_readout/: PYTHONPATH=paper python paper/make_tables_g6.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_tables import R, T
from make_tables_g4 import MODEL, ORDER
from make_tables_g5 import INK, MUTED, BLUE, ORANGE, p
from make_figures_g4 import zero_label, icl_gains

DEPTHS = ["0", "12.5", "25", "50", "75"]
SETNAME = {"illusions": "Illusions (VLMBias)", "illusynC": "Illusions (synthetic)", "countbench": "CountBench",
           "dice": "Dice", "tally": "Tally", "carlogos": "Car logos"}


def swap():
    d = pd.read_csv(f"{R}/g6/swap_delta.csv").set_index("model")
    x = [float(t) for t in DEPTHS]
    fig, ax = plt.subplots(figsize=(3.3, 1.85))
    for what, col, lab in (("probe", BLUE, "decision-state probe"), ("answer", ORANGE, "model's answer")):
        Y = np.array([[d.loc[m, f"D_{what}@{t}"] for t in DEPTHS] for m in d.index]) * 100
        for row in Y:
            ax.plot(x, row, color=col, lw=0.6, alpha=0.35)
        ax.plot(x, np.nanmedian(Y, 0), color=col, lw=2, marker="o", ms=3, label=lab)
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}%" for t in DEPTHS])
    ax.set_xlabel("depth of the swap (share of LLM layers)", color=INK)
    ax.set_ylabel("label-specific effect (pts)", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout(pad=0.2)
    fig.savefig("paper/fig/swap.pdf"), fig.savefig("paper/fig/swap.png", dpi=200)
    rows = []
    for m in sorted(d.index, key=ORDER.index):
        r = d.loc[m]
        rows.append(f"{MODEL[m]}{'$^*$' if r.unseen else ''} & " + " & ".join(f"{100 * r[f'D_probe@{t}']:+.0f}" for t in DEPTHS) + " & "
                    + " & ".join(f"{100 * r[f'D_answer@{t}']:+.0f}" for t in DEPTHS) + r" \\")
    med = lambda w, t: f"{100 * d[f'D_{w}@{t}'].median():+.0f}"
    rows.append(r"\midrule" + "\nMedian & " + " & ".join(med("probe", t) for t in DEPTHS) + " & " + " & ".join(med("answer", t) for t in DEPTHS) + r" \\")
    open(f"{T}/supp_swapdelta.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Label-specific interchange effect} (G6 M$'$), in points: share of base items on which the read-out takes the partner's label after an other-label swap, minus the share on which it takes the other label after a same-label swap, at the input of the layer at the given depth (prompt pass only; depth 0 = running the partner). $^*$: models whose swap results were not seen before the G6 protocol.}
\label{tab:supp_swapdelta}
\begin{tabular}{@{}l""" + "r" * 10 + r"""@{}}
\toprule
& \multicolumn{5}{c}{Decision-state probe} & \multicolumn{5}{c}{Model's answer} \\
\cmidrule(lr){2-6}\cmidrule(l){7-11}
Model & """ + " & ".join([f"{t}\\%" for t in DEPTHS] * 2) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    print("swap delta medians", {w: [round(100 * d[f"D_{w}@{t}"].median()) for t in DEPTHS] for w in ("probe", "answer")}, "n", len(d))


def price():
    P = pd.read_csv(f"{R}/g6/price.csv")
    S = pd.read_csv(f"{R}/explore_disguise.csv")
    fmt = lambda v: "$>$32" if not np.isfinite(v) else f"{v:g}"
    icl = icl_gains(ks=(8,)).get(8, [])
    free = {s: zero_label(s) for s in ("illusions", "countbench")}
    rows = []
    for s in ["countbench", "dice", "tally", "carlogos", "illusions", "illusynC"]:
        c = P[P.set == s]
        if c.empty:
            continue
        sh = lambda t: f"{100 * c[f'share_{t}'].median():.0f}"
        sep = f"{100 * (c.share_INHERITED + c.share_DROPPED).median():.0f}"
        pr = " & ".join(fmt(c[f"price_{f}"].quantile(0.5, interpolation="higher")) for f in ("cal", "inp", "dec", "enc"))
        st = S[(S.set == s) & (S.target == "own")]
        steer = f"{100 * (st.acc - st.raw).median():+.0f}" if len(st) else "--"
        think = f"{free[s]['R4']:+.1f}" if s in free else "--"
        ic = f"{np.mean(icl):+.1f}" if s == "illusions" and icl else "--"
        rows.append(f"{SETNAME[s]} & {len(c)} & {sh('BIAS')} & {sep} & {sh('UNSEEN')} & {pr} \\\\")
        print(s, "steer", steer, "think", think, "icl", ic)
    open(f"{T}/price.tex", "w").write(r"""\begin{table}[t]
\centering\footnotesize
\caption{\textbf{The price of a repair.} Error types on one half of the base images (median share of errors, \%; $E$ = own LLM input) and, on the other half, the labelled items per class at which a readout first beats the model by 5 points (median over models with $\geq$30 errors; $>$32: never within 32). Cal: candidate logits; inp: own LLM input; dec: decision state; enc: SigLIP2. Dice and car logos: VLMBias subsets whose answers the models never give.}
\label{tab:price}
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
& & \multicolumn{3}{c}{Types (\%)} & \multicolumn{4}{c}{Price (labels/class)} \\
\cmidrule(lr){3-5}\cmidrule(l){6-9}
Benchmark & $n$ & \textsc{b} & \textsc{i+d} & \textsc{u} & cal & inp & dec & enc \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    print("free", {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in free.items()}, "icl8", np.mean(icl) if icl else None)
    rows = []
    for s in ["countbench", "dice", "tally", "carlogos", "illusions", "illusynC"]:
        c = P[P.set == s].assign(o=lambda z: z.model.map(lambda m: ORDER.index(m) if m in ORDER else 99)).sort_values("o")
        if c.empty:
            continue
        rows.append((r"\midrule" + "\n" if rows else "") + rf"\multicolumn{{10}}{{@{{}}l}}{{\emph{{{SETNAME[s]}}}}} \\")
        for r in c.itertuples():
            rows.append(f"{MODEL.get(r.model, r.model)} & {r.errors_A} & {100 * r.share_BIAS:.0f} & {100 * (r.share_INHERITED + r.share_DROPPED):.0f} & "
                        f"{100 * r.share_UNSEEN:.0f} & {p(r.raw_B)} & " + " & ".join(fmt(getattr(r, f"price_{f}")) for f in ("cal", "inp", "dec", "enc")) + r" \\")
    open(f"{T}/supp_price.tex", "w").write(r"""\begin{table*}[t]
\centering\scriptsize
\caption{\textbf{Price of repair per cell} (G6 P). Types on half A of the base images ($E$ = own LLM input; errors on A, shares in \%), prices on half B (labelled items per class at which the ridge readout first beats the model by 5 points; mean over 30 draws; raw: model accuracy on B, \%).}
\label{tab:supp_price}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
Model & errors & \bias & \inherited{+}\dropped & \unseen & raw & calibrator & own input & decision & SigLIP2 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


if __name__ == "__main__":
    import os
    if os.path.exists(f"{R}/g6/swap_delta.csv"):
        swap()
    if os.path.exists(f"{R}/g6/price.csv"):
        price()
