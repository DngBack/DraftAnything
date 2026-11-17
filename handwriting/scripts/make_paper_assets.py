"""Build the paper's tables, numbers and figure directly from reports/acl, so nothing is typed by hand."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, P = Path("reports/acl"), Path("paper")
SETS = [("train", "Discovery"), ("val1", "Replication"), ("confirm", "Confirmation")]
NAMES = {"surprisal": "Surprisal", "tree_dist": "Tree distance", "clause_start": "Clause onset"}
INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def f(x, n=3):
    return f"{x:.{n}f}".replace("-", "$-$")


def p_str(x):
    return "$<$.001" if x < 0.001 else f"{x:.3f}".lstrip("0")


tabs = {s: pd.read_csv(R / f"robust_{s}.csv") for s, _ in SETS}
ext = json.loads((R / "extensions.json").read_text())
prim = {s: t[(t.spec == "primary") & (t.cluster == "writer")].set_index("predictor") for s, t in tabs.items()}

# Table 1
rows = [r"\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & Surprisal & Tree distance & Clause onset \\", r"\midrule"]
for s, name in SETS:
    t = prim[s]
    rows.append(name + " & " + " & ".join(f"{f(t.loc[p, 'beta_per_sd'])}" for p in NAMES) + r" \\")
    rows.append(r"\quad 95\% CI & " + " & ".join(f"[{f(t.loc[p, 'ci_lo'])}, {f(t.loc[p, 'ci_hi'])}]" for p in NAMES) + r" \\")
    rows.append(r"\quad Holm $p$ & " + " & ".join(p_str(t.loc[p, "p_holm"]) for p in NAMES) + r" \\[2pt]")
pp = ext["pooled"]["primary"]
rows.append("Pooled & " + " & ".join(f(pp[p]["beta"]) for p in NAMES) + r" \\")
rows.append(r"\quad 95\% CI & " + " & ".join(f"[{f(pp[p]['ci'][0])}, {f(pp[p]['ci'][1])}]" for p in NAMES) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
(P / "table_main.tex").write_text("\n".join(rows) + "\n")

# Table 2
SPEC = [("primary", "Primary"), ("lower", "Lowercase words only"), ("pos", "+ LOB tags"), ("gbm", "GBM controls"), ("clip", "Gap clipped 1--99\\%"), ("pythia", "Pythia surprisal")]
rows = [r"\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Specification & Surprisal & Tree dist. & Clause onset \\", r"\midrule"]
for k, label in SPEC:
    d = ext["pooled"][k]
    keys = ["surprisal_alt" if k == "pythia" else "surprisal", "tree_dist", "clause_start"]
    cell = lambda q: (r"\textbf{" if d[q]["ci"][0] > 0 or d[q]["ci"][1] < 0 else "{") + f(d[q]["beta"]) + "}"
    rows.append(label + " & " + " & ".join(cell(q) for q in keys) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
(P / "table_robust.tex").write_text("\n".join(rows) + "\n")

# Numbers used in the prose
t1 = {o["split"]: o for o in json.loads((R / "t1_train_val1.json").read_text())}
pc = json.loads((R / "printed_control.json").read_text())
hets = {s: json.loads((R / f"hetero_{s}.json").read_text()) for s, _ in SETS}
macros = {}
m = lambda k, v: macros.__setitem__(k, v)
for s, tag in [("train", "Disc"), ("val1", "Rep"), ("confirm", "Conf")]:
    t = prim[s]
    m(f"n{tag}", f"{int(t.n.iloc[0]):,}")
    m(f"w{tag}", str(int(t.n_writers.iloc[0])))
    m(f"hetS{tag}", f"{100 * hets[s]['surprisal']['frac_writers_positive']:.0f}")
for s, tag in [("train", "Disc"), ("val1", "Rep")]:
    o = t1[s]
    m(f"agr{tag}", f(o["same_boundary"]["r"]))
    m(f"agr{tag}CI", f"[{f(o['same_boundary']['ci'][0])}, {f(o['same_boundary']['ci'][1])}]")
    m(f"oth{tag}", f(o["same_pair_other_prompt"]["r"]))
    m(f"oth{tag}CI", f"[{f(o['same_pair_other_prompt']['ci'][0])}, {f(o['same_pair_other_prompt']['ci'][1])}]")
    m(f"nul{tag}", f"[{f(o['null_within_form_shuffle']['p2.5_97.5'][0])}, {f(o['null_within_form_shuffle']['p2.5_97.5'][1])}]")
    m(f"pairs{tag}", f"{o['same_boundary']['n_pairs']:,}")
m("agrNoPrint", f(pc["agreement_no_printed"]["r"]))
m("agrPrint", f(pc["agreement_with_printed"]["r"]))
m("agrPrintCI", f"[{f(pc['agreement_with_printed']['ci'][0])}, {f(pc['agreement_with_printed']['ci'][1])}]")
b = pc["pooled"]
m("printAlone", f(b["printed_alone|print_rel_f"]["beta"]))
m("printAloneCI", f"[{f(b['printed_alone|print_rel_f']['ci'][0])}, {f(b['printed_alone|print_rel_f']['ci'][1])}]")
m("sSub", f(b["without_printed|surprisal"]["beta"]))
m("sSubP", f(b["with_printed|surprisal"]["beta"]))
m("sdGap", f(ext["within_line_sd_gap_train"], 2))
m("sdBg", f(ext["within_line_sd_bg_train"], 1))
pl = tabs["train"][(tabs["train"].spec == "placebo_bg") & (tabs["train"].cluster == "writer")].set_index("predictor")
m("placeboT", f(pl.loc["tree_dist", "beta_per_sd"]))
m("placeboTp", p_str(pl.loc["tree_dist", "p_holm"]))
fu = json.loads((R / "followups.json").read_text())
for key, tag in [("all|primary|clause_start", "ClauseFive"), ("all|A_strict_clause|clause_strict", "ClauseStrict"),
                 ("all|B_context|surprisal", "Ctx"), ("all|B_context|surprisal_nc", "Nc"),
                 ("lower|B_context|surprisal", "CtxLow")]:
    for scope, suffix in [("all_three", ""), ("val1_confirm", "Held")]:
        v = fu[key][scope]
        m(f"fu{tag}{suffix}", f(v["beta"]))
        m(f"fu{tag}{suffix}CI", f"[{f(v['ci_lo'])}, {f(v['ci_hi'])}]")
(P / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
for k, v in macros.items():
    print(k, v)

# Figure: effect estimates by writer set; single-hue dot-and-interval small multiples.
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(3.3, 1.55), sharey=True, sharex=True)
for ax, (p, title) in zip(axes, NAMES.items()):
    ys = [3, 2, 1]
    for y, (s, _) in zip(ys, SETS):
        r = prim[s].loc[p]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=BLUE, lw=1.4, solid_capstyle="round")
        ax.plot(r.beta_per_sd, y, "o", ms=4, mfc=BLUE, mec="white", mew=0.6)
    q = pp[p]
    ax.plot(q["ci"], [0, 0], color=INK, lw=1.4, solid_capstyle="round")
    ax.plot(q["beta"], 0, "D", ms=3.8, mfc=INK, mec="white", mew=0.6)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_title(title, fontsize=8, color=INK, pad=3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 0.07)
    ax.set_xticks([0, 0.03, 0.06])
    ax.set_xticklabels(["0", ".03", ".06"])
axes[0].set_yticks([3, 2, 1, 0])
axes[0].set_yticklabels(["Discovery", "Replication", "Confirmation", "Pooled"])
axes[1].set_xlabel("Added gap (core heights) per SD", fontsize=7.5)
fig.subplots_adjust(left=0.255, right=0.975, top=0.88, bottom=0.26, wspace=0.09)
fig.savefig(P / "fig_effects.pdf")
fig.savefig(P / "fig_effects.png", dpi=170)
