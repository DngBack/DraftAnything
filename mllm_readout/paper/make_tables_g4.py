"""G4 tables (input-side control, repair matrix, probe-in-disguise, label cost).
Run from mllm_readout/: python paper/make_tables_g4.py"""
import glob
import numpy as np
import pandas as pd
from make_tables import MODEL as M0, R, T

MODEL = {**M0, "qwen2vl_2b": "Qwen2-VL-2B", "qwen35_2b": "Qwen3.5-2B", "qwen35_4b": "Qwen3.5-4B", "qwen35_9b": "Qwen3.5-9B"}
ORDER = list(MODEL)
TYPES = ["BIAS", "UNSPOKEN", "INHERITED", "DROPPED", "UNSEEN"]
TNAME = {"BIAS": r"\textsc{bias}", "UNSPOKEN": r"\textsc{unspoken}", "INHERITED": r"\textsc{inherited}",
         "DROPPED": r"\textsc{dropped}", "UNSEEN": r"\textsc{unseen}", "LOSS": r"\textsc{loss}"}
G4 = ["qwen35_2b", "qwen35_4b", "qwen35_9b", "qwen3vl4b", "qwen3vl8b"]


def med(x, pct=True):
    x = pd.Series(x).dropna()
    f = (lambda v: f"{100 * v:.0f}") if pct else (lambda v: "0" if round(v) == 0 else f"{v:+.0f}".replace("-", "$-$"))
    return f"{f(x.median())} {{\\scriptsize[{f(x.min())}--{f(x.max())}]}}"


def types():
    d = pd.read_csv(f"{R}/g4_types.csv")
    va = pd.read_csv(f"{R}/explore_valueadd.csv")
    cols = {}
    for s in ("illusions", "countbench"):
        x = d[d.set == s]
        v = va[va.set == s]
        err = x[TYPES].sum(axis=1)
        cols[s] = [
            med(x.raw), med(x.acc_cal), med(x.acc_ans), med(x.acc_mllmL), med(x.acc_mllm0), med(x.acc_ext_mean), med(x.acc_ext),
            med(x.acc_dino_mean), med(x.acc_ext_shuf),
            med(x.acc_loso_ans) if s == "illusions" else "--", med(x.acc_loso_ext) if s == "illusions" else "--",
            med(100 * v.llm_adds, False), med(100 * v.input_adds, False),
        ] + [med(x[t] / err) for t in TYPES]
        cols[s + "_n"] = len(x)
    names = ["Model output", "Calibrator (own logits)", "Decision state, 75\\% depth", "Image tokens, 75\\% depth",
             "Image tokens, layer 0", "SigLIP2 (frozen), mean", "SigLIP2, 4$\\times$4 grid (types)", "DINOv2 (frozen)", "SigLIP2, shuffled labels",
             "Decision state, leave-one-type-out", "SigLIP2, leave-one-type-out",
             "LLM adds over SigLIP2 (pts)", "SigLIP2 adds over LLM (pts)"] + [TNAME[t] for t in TYPES]
    rows = []
    for i, n in enumerate(names):
        if i == 0:
            rows.append(r"\multicolumn{3}{@{}l}{\emph{Accuracy of each readout (\%)}} \\")
        if i == 11:
            rows.append(r"\multicolumn{3}{@{}l}{\emph{Value added, probe on the concatenation}} \\")
        if i == 13:
            rows.append(r"\multicolumn{3}{@{}l}{\emph{Share of errors (\%)}} \\")
        rows.append(f"{n} & {cols['illusions'][i]} & {cols['countbench'][i]} \\\\")
    open(f"{T}/types.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{Who is doing the seeing?} Median [min--max] over """ + str(cols["illusions_n"]) + r""" models of linear readouts fitted with 5-fold cross-fitting grouped by base image. On the illusions, a frozen image encoder separates the answers better than the MLLM's own decision state and the LLM adds nothing to it; on CountBench, the count is computed by the LLM. Leave-one-type-out: trained on five illusion types, tested on the sixth. Error types: \cref{sec:method}.}
\label{tab:types}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrr@{}}
\toprule
& Illusions & CountBench \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # supplementary: per model
    rows = []
    for s in ("illusions", "countbench"):
        x = d[d.set == s].assign(o=lambda z: z.model.map(ORDER.index)).sort_values("o")
        rows.append(r"\midrule" if rows else "")
        rows.append(rf"\multicolumn{{14}}{{@{{}}l}}{{\emph{{{'Illusions' if s == 'illusions' else 'CountBench'}}}}} \\")
        for r in x.itertuples():
            f = lambda v: "--" if pd.isna(v) else f"{100 * v:.0f}"
            rows.append(f"{MODEL[r.model]} & {r.errors} & {f(r.raw)} & {f(r.acc_cal)} & {f(r.acc_ans)} & {f(r.acc_mllmL)} & "
                        f"{f(r.acc_mllm0)} & {f(r.acc_ext_mean)} & " + " & ".join(str(int(getattr(r, t))) for t in TYPES) +
                        f" & {f(r.loss_ext)} \\\\")
    open(f"{T}/supp_types.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Bracketed diagnostic per model} (G4; accuracies in \%, error counts). Img.\,75\%/L0: mean image-token state at 75\% depth / projector output. SigLIP2: frozen encoder, mean-pooled. Last column: share of \loss errors (calibrator and model wrong, decision-state probe right) on which the SigLIP2 probe is right (I3; shuffled-label control 42--53\%).}
\label{tab:supp_types}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrrrrrr@{}}
\toprule
Model & \#err & Raw & Cal. & Dec.\,75\% & Img.\,75\% & Img.\,L0 & SigLIP2 & \textsc{bias} & \textsc{unsp.} & \textsc{inh.} & \textsc{drop.} & \textsc{unseen} & I3 \\
""" + "\n".join(r for r in rows if r) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def matrix():
    D = pd.read_csv(f"{R}/g4_matrix_items.csv")
    cols = ["R1", "R2", "R3", "R3x", "R4", "R5"]
    diag = {"BIAS": "R1", "UNSPOKEN": "R2", "INHERITED": "R4", "UNSEEN": "R5"}
    err = D[D.type.isin(TYPES)]
    rows = []
    for t in TYPES:
        g = err[err.type == t]
        cells = []
        for c in cols:
            v, n = g[c].mean(), g[c].notna().sum()
            s = f"{100 * v:.0f}"
            best = v >= g[cols].mean().max() - 1e-9
            s = rf"\textbf{{{s}}}" if best else s
            s = rf"\fbox{{{s}}}" if diag.get(t) == c else s
            cells.append(s)
        rows.append(f"{TNAME[t]} ({len(g)}) & " + " & ".join(cells) + r" \\")
    ok = D[D.type == "CORRECT"]
    rows.append(r"\midrule")
    rows.append("Broken (correct$\\to$wrong) & " + " & ".join(f"{100 * (1 - ok[c].mean()):.0f}" for c in cols) + r" \\")
    open(f"{T}/matrix.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{Repair matrix} (pre-registered; 5 models $\times$ illusions, CountBench, MMVP). Share of errors of each type fixed by each repair (\%; bold: best in row). Boxed: the repair pre-assigned to the row (P1; R3/R3x excluded from P1). Labels: how many labelled items the repair uses; ``fold'' = all items outside the test fold (4/5 of the set, 240--634). R4 is run on Qwen3.5 only, R3x not on MMVP.}
\label{tab:matrix}
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& R1 & R2 & R3 & R3x & R4 & R5 \\
& calib. & guid. & steer & steer$_\text{SigLIP}$ & think & 2$\times$ \\
Labels & fold & 0 & fold & fold & 0 & 0 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # supplementary: accuracy per cell
    D["G3"] = (D.type == "CORRECT").astype(float)
    acc = D.groupby(["model", "set"])[["G3"] + cols + ["R6x", "plain"]].mean()
    rows, last = [], None
    for (m, s), r in acc.iterrows():
        f = lambda v: "--" if pd.isna(v) else f"{100 * v:.1f}"
        rows.append(f"{MODEL[m] if m != last else ''} & {s.replace('countbench', 'CountBench').replace('illusions', 'Illusions').replace('mmvp', 'MMVP')} & "
                    + " & ".join(f(r[c]) for c in ["G3"] + cols + ["plain", "R6x"]) + r" \\")
        last = m
    open(f"{T}/supp_matrix.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Repair accuracy per cell} (\%; illusions: one view per image and wording, 264 items; CountBench 437; MMVP 300). G3: stored decision of the unedited model. Plain: free generation with the same code path as R6x (question before image; exploratory).}
\label{tab:supp_matrix}
\begin{tabular}{@{}llrrrrrrrrr@{}}
\toprule
Model & Set & G3 & R1 calib. & R2 guid. & R3 steer & R3x steer$_\text{SigLIP}$ & R4 think & R5 2$\times$ & Plain & R6x q-first \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def disguise():
    d = pd.read_csv(f"{R}/explore_disguise.csv")
    rows = []
    for s in ("illusions", "countbench"):
        x = d[d.set == s]
        rows.append(r"\midrule" if rows else "")
        rows.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{'Illusions' if s == 'illusions' else 'CountBench'}}}}} \\")
        for m in sorted(x.model.unique(), key=ORDER.index):
            y = x[x.model == m].set_index("target")
            f = lambda t: f"{100 * y.loc[t, 'acc']:.0f} {{\\scriptsize({100 * y.loc[t, 'W']:.0f})}}" if t in y.index else "--"
            rows.append(f"{MODEL[m]} & {100 * y.acc.iloc[0] if False else 100 * y.raw.iloc[0]:.0f} & " +
                        " & ".join(f(t) for t in ["own", "ext", "oracle", "anti", "random", "shuffle"]) + r" \\")
    open(f"{T}/supp_disguise.tex", "w").write(r"""\begin{table*}[t]
\centering\small
\caption{\textbf{What steering writes} ($\alpha{=}16$, class-mean directions of the MLLM's decision state, cross-fitted by image). Accuracy after steering (\%) and, in brackets, writeability $W$ = share of edited items whose answer becomes the target. Targets: own = the MLLM decision-state probe; SigLIP2 = a probe on the frozen encoder; oracle = the true class; anti = a wrong class; random = a random direction of the same norm; shuffle = class means from shuffled labels.}
\label{tab:supp_disguise}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Model & Raw & Own probe & SigLIP2 probe & Oracle & Anti & Random dir. & Shuffled \\
""" + "\n".join(r for r in rows if r) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    t = d[d.target.isin(["own", "ext", "oracle", "anti"])]
    print("disguise: corr(acc, target acc) =", round(np.corrcoef(t.acc, t.probe_acc)[0, 1], 3), "n =", len(t))


def labelcost():
    D = pd.read_csv(f"{R}/explore_labelcost.csv")
    x = D[(D.set == "illusions")].groupby(["model", "k"])[["raw", "cal_ridge", "ans_ridge", "mllm0_ridge", "ext_ridge"]].mean()
    rows = []
    for m in sorted(x.index.get_level_values(0).unique(), key=ORDER.index):
        r = x.loc[m]
        f = lambda c, k: "--" if pd.isna(r.loc[k, c]) else f"{100 * r.loc[k, c]:.0f}"
        rows.append(f"{MODEL[m]} & {100 * r.raw.mean():.0f} & " + " & ".join(f"{f(c, 8)}/{f(c, 32)}" for c in
                                                                         ["cal_ridge", "ans_ridge", "mllm0_ridge", "ext_ridge"]) + r" \\")
    icl = {}
    for f in glob.glob(f"{R}/g4/icl_*_k*.npz"):
        m, k = f.split("icl_")[1][:-4].rsplit("_k", 1)
        z = np.load(f)
        icl[(m, int(k))] = 100 * np.mean(z["pred"] == z["y"])
    irows = [f"{MODEL[m]} & " + " & ".join(f"{icl[(m, k)]:.1f}" for k in (1, 2, 4, 8)) + r" \\"
             for m in sorted({m for m, _ in icl}, key=ORDER.index)]
    open(f"{T}/supp_labelcost.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{\textbf{Label cost on the illusions, per model.} Ridge readouts with $k{=}8$ / $k{=}32$ labelled items per class and question wording, trained on half of the base images and tested on the other half (30 draws; \%). The SigLIP2 column is the same encoder for every model. Bottom: in-context learning with $k$ labelled demonstrations per class of the same wording (other base images), 264 test items.}
\label{tab:supp_labelcost}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Model & Raw & Calib. & Dec.\ state & Img.\ L0 & SigLIP2 \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
ICL & $k{=}1$ & 2 & 4 & 8 & \\
\midrule
""" + "\n".join(r.replace(r" \\", r" & \\") for r in irows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


if __name__ == "__main__":
    types()
    matrix()
    disguise()
    labelcost()
