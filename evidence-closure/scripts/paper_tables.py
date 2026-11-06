"""Every number in the paper comes from here: results/*.jsonl -> paper/numbers.tex, paper/tables/*.tex, paper/fig/*.pdf.

Protocol verdicts apply the locked criteria of docs/protocol_decide.md and docs/protocol_token.md. D4a's three parts are all required at
both s40 and s30; the protocol is not explicit for the discordant-pair part, and this is the stricter reading.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_attr as AT  # noqa: E402
import analyze_decide as AD  # noqa: E402
import analyze_token as AK  # noqa: E402
from evc.degrade import LEVELS  # noqa: E402

PAPER = ROOT / "paper"
TAGS = {"qwen25vl7b": ("seven", "Qwen2.5-VL-7B"), "qwen3vl8b": ("three", "Qwen3-VL-8B"), "internvl3": ("intern", "InternVL3-8B")}
COLOR = {"qwen25vl7b": "#2a78d6", "qwen3vl8b": "#eb6834", "internvl3": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
V = {}  # macro key -> text


def pm(e, ci, k=100):
    return f"{k * e:+.1f} [{k * ci[0]:+.1f}, {k * ci[1]:+.1f}]"


def put(key, text):
    V[key] = str(text).replace("%", r"\%")


def write_table(name, lines):
    (PAPER / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ---------- Protocol D ----------
def decide(tag):
    S, rows = AD.load("decide", tag, "decide_set.json")
    count = Counter(r["qid"] for r in rows)
    rows = [r for r in rows if count[r["qid"]] == 56]  # the run stopped early: keep only cells with every condition
    t = AD.Table(S, rows)
    g = t.g
    mean_ci = lambda x: (np.nanmean(x), t.ci(lambda i: np.nanmean(x[i])))  # noqa: E731
    R = dict(n=len(t.qs), lv={})
    to, tn, tc = g("T", 1, "s100", "orig", "copy"), g("T", 1, "s100", "null", "copy"), g("T", 1, "s100", "cf", "copy")
    bo = np.nanmean(g("T", 3, "s100", "decoy", "boxonly"))
    R["valid"] = dict(null=np.nanmean(tn - to), cf=np.nanmean(tc), orig=np.nanmean(to), box=bo, box40=np.nanmean(g("T", 3, "s40", "decoy", "boxonly")),
                      ok=bool(abs(np.nanmean(tn - to)) <= 0.05 and np.nanmean(tc) >= 0.80 and bo >= 0.70))
    R["hijack"] = {lv: (np.nanmean(g("T", 3, lv, "decoy", "boxcopy")), np.nanmean(g("V", 3, lv, "decoy", "boxcopy"))) for lv in ("s100", "s40")}
    bl = [r for r in rows if r["cond"] == "blank" and r["prompt"] == "copy"]
    R["blank"] = dict(number=np.mean([bool(AD.nums(r["text"])) for r in bl]), refuse=np.nanmean(g("U", 3, "s100", "blank", "copyu2")))
    for lv in ("s100", "s50", "s40", "s30"):
        x = dict(orig=np.nanmean(g("T", 1, lv, "orig", "copy")), cf=np.nanmean(g("T", 1, lv, "cf", "copy")))
        d = g("T", 1, lv, "orig", "copy") - g("T", 1, lv, "orig", "addr")
        x["task"] = mean_ci(d)
        d = g("T", 1, lv, "orig", "copy") - g("T", 1, lv, "orig", "program")
        x["prog"] = mean_ci(d)
        if lv != "s100":
            d = g("O", 1, lv, "cf", "copy") - g("A", 1, lv, "cf", "copy")
            x["pull"] = mean_ci(d)
            x["O"] = np.nanmean(g("O", 1, lv, "cf", "copy"))
            b, l, gg, lg = g("T", 2, lv, "base", "copy"), g("T", 2, lv, "L", "boxcopy"), g("T", 2, lv, "G", "copy"), g("T", 2, lv, "LG", "boxcopy")
            x.update(base=np.nanmean(b), L=np.nanmean(l), G=np.nanmean(gg), LG=np.nanmean(lg))
            x["phiL"] = mean_ci(((l - b) + (lg - gg)) / 2)
            x["phiG"] = mean_ci(((gg - b) + (lg - l)) / 2)
            x["inter"] = mean_ci(lg - l - gg + b)
            x["upscale"] = mean_ci(b - g("T", 1, lv, "orig", "copy"))
            # exploratory simple effects of the box, with and without restored pixels
            x["boxBlur"], x["boxSharp"] = mean_ci(l - b), mean_ci(lg - gg)
        y, lp = g("T", 1, lv, "orig", "copy"), g("lp", 1, lv, "orig", "copy")
        x["au"] = AD.au(lp, y)
        x["au_ci"] = t.ci(lambda i: AD.au(lp, y, i))  # noqa: B023
        x["au_size"] = AD.au(t.h, y)
        x["au_diff"] = (x["au"] - x["au_size"], t.ci(lambda i: AD.au(lp, y, i) - AD.au(t.h, y, i)))  # noqa: B023
        conds = [("orig", "copy"), ("null", "copy"), ("cf", "copy")]
        Y, P = np.stack([g("T", 1, lv, *c) for c in conds], 1), np.stack([g("lp", 1, lv, *c) for c in conds], 1)
        per = np.full(len(t.qs), np.nan)
        for q in range(len(t.qs)):
            w = [P[q, u] - P[q, v] for u in range(3) for v in range(3) if Y[q, u] == 1 and Y[q, v] == 0]
            if w:
                per[q] = np.mean(np.array(w) > 0)
        x["disc"] = mean_ci(per)
        x["disc_n"] = int((~np.isnan(per)).sum())
        ref = {}
        for p in ("copyu", "copyu2"):
            u = g("U", 1, lv, "orig", p)

            def dif(i, u=u, y=y):
                return np.nanmean(u[i][y[i] == 0]) - np.nanmean(u[i][y[i] == 1])
            ref[p] = dict(rate=np.nanmean(u), wrong=np.nanmean(u[y == 0]), right=np.nanmean(u[y == 1]), gap=(dif(np.arange(len(y))), t.ci(dif)))
        x["ref"] = ref
        yc, lc, c = g("T", 1, lv, "orig", "conf"), g("lp", 1, lv, "orig", "conf"), g("C", 1, lv, "orig", "conf")
        have = ~np.isnan(c)
        x["conf"] = dict(au_lp=AD.au(lc[have], yc[have]), au_c=AD.au(c, yc), mean=np.nanmean(c), have=have.mean(), out=int(np.nansum(c > 100)))
        x["conf"]["diff"] = (x["conf"]["au_lp"] - x["conf"]["au_c"], t.ci(lambda i: (lambda j: AD.au(lc, yc, j) - AD.au(c, yc, j))(i[have[i]])))  # noqa: B023
        ok = have & (c <= 100)  # sensitivity only; the locked scoring keeps out-of-scale values
        x["conf"]["au_c_inscale"] = AD.au(c[ok], yc[ok])
        R["lv"][lv] = x
    lv = R["lv"]
    # locked verdicts
    pos = lambda e: e[1][0] > 0  # noqa: E731
    R["D1"] = [k for k in ("s50", "s40", "s30") if pos(lv[k]["phiL"]) and pos(lv[k]["phiG"])]
    p40, p30 = lv["s40"]["pull"], lv["s30"]["pull"]
    has = any(p[1][0] > 0 and p[0] >= 0.03 for p in (p40, p30))
    R["D2"] = "yes" if has else "negligible" if (p40[1][1] < 0.03 and p30[1][1] < 0.03) else "open"
    ts = [lv[k]["task"] for k in ("s50", "s40", "s30")]
    has = any((e[1][0] > 0 or e[1][1] < 0) and abs(e[0]) >= 0.05 for e in ts)
    R["D3"] = "yes" if has else "negligible" if all(e[1][0] > -0.05 and e[1][1] < 0.05 for e in ts) else "open"
    R["D4a"] = bool(all(lv[k]["au"] >= 0.75 and pos(lv[k]["au_diff"]) and lv[k]["disc"][1][0] > 0.5 for k in ("s40", "s30")))
    gaps = [lv["s30"]["ref"][p]["gap"] for p in ("copyu", "copyu2")]
    R["D4b"] = bool(any(e[0] >= 0.20 and e[1][0] > 0 for e in gaps))
    R["D4c"] = bool(any(pos(lv[k]["conf"]["diff"]) for k in ("s40", "s30")))
    return R


def macros_decide(key, R):
    put(f"{key}.n", R["n"])
    put(f"{key}.null", f"{100 * R['valid']['null']:+.1f}")
    put(f"{key}.cf", f"{100 * R['valid']['cf']:.1f}")
    put(f"{key}.box", f"{100 * R['valid']['box']:.1f}")
    put(f"{key}.blank", f"{100 * R['blank']['number']:.1f}")
    put(f"{key}.blankref", f"{100 * R['blank']['refuse']:.1f}")
    for k, (a, b) in R["hijack"].items():
        put(f"{key}.{k}.hijack", f"{100 * a:.1f}")
        put(f"{key}.{k}.hijacktarget", f"{100 * b:.1f}")
    for k, x in R["lv"].items():
        put(f"{key}.{k}.orig", f"{100 * x['orig']:.1f}")
        put(f"{key}.{k}.au", f"{x['au']:.2f}")
        put(f"{key}.{k}.auc", f"{x['conf']['au_c']:.2f}")
        put(f"{key}.{k}.task", pm(*x["task"]))
        put(f"{key}.{k}.prog", pm(*x["prog"]))
        put(f"{key}.{k}.disc", pm(x["disc"][0], x["disc"][1]))
        put(f"{key}.{k}.refrate", f"{100 * x['ref']['copyu2']['rate']:.1f}")
        put(f"{key}.{k}.refgap", pm(*x["ref"]["copyu2"]["gap"]))
        put(f"{key}.{k}.confout", x["conf"]["out"])
        put(f"{key}.{k}.confmean", f"{x['conf']['mean']:.1f}")
        put(f"{key}.{k}.aucin", f"{x['conf']['au_c_inscale']:.2f}")
        if k != "s100":
            for m in ("base", "L", "G", "LG", "O"):
                put(f"{key}.{k}.{m}", f"{100 * x[m]:.1f}")
            for m in ("phiL", "phiG", "inter", "upscale", "pull", "boxBlur", "boxSharp"):
                put(f"{key}.{k}.{m}", pm(*x[m]))


# ---------- ATTR ----------
def attr(tag):
    data = {}
    for lv in LEVELS:
        p = ROOT / "results" / f"attr_{tag}_{lv}.jsonl"
        if p.exists():
            data[lv] = {r["qid"]: r for r in map(json.loads, open(p))}
    common = sorted(set.intersection(*[set(d) for d in data.values()]))
    W = {lv: {c: np.array([AT.ev(d[q], *c) for q in common]) for c in AT.COAL} for lv, d in data.items()}
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(common), len(common)) for _ in range(1000)]  # one question per document in this set
    allq = np.arange(len(common))
    B = {lv: [AT.stats(W[lv], b) for b in boots] for lv in W}
    R = dict(n=len(common), lv={})
    for lv in W:
        s, sh, b0, b1 = AT.stats(W[lv], allq)
        ci = np.percentile([x[0] for x in B[lv]], [2.5, 97.5], axis=0)
        sci = np.percentile([x[1][0] for x in B[lv]], [2.5, 97.5])
        R["lv"][lv] = dict(phi=s, ci=ci, share=sh[0], share_ci=sci, em=b0, top=b1)
    d = [x[1][0] - y[1][0] for x, y in zip(B["s30"], B["s100"])]
    R["dshare"] = (R["lv"]["s30"]["share"] - R["lv"]["s100"]["share"], np.percentile(d, [2.5, 97.5]))
    s = R["lv"]["s100"]["phi"]
    d3 = [x[0][1] + x[0][2] - x[0][0] for x in B["s100"]]
    R["crit3"] = (s[1] + s[2] - s[0], np.percentile(d3, [2.5, 97.5]))
    return R


# ---------- Protocol T ----------
def token(tag):
    qs, T, LP, SZ = AK.load(tag)
    if len(qs) < 200 or any(np.isnan(v).any() for v in T.values()):  # incomplete run: nothing goes into the paper yet
        return None
    R = dict(n=len(qs), T={f"{k[0]}|{k[1]}": np.nanmean(v) for k, v in T.items()}, size={f"{k[0]}|{k[1]}": np.nanmean(v) for k, v in SZ.items()})
    conv = lambda e: (e[0] / 100, (e[1] / 100, e[2] / 100))  # noqa: E731
    if tag == "internvl3":
        R["t3"] = conv(AK.paired(T[("s100", "tile12")], T[("s100", "tile1")]))
        R["t4"] = {lv: conv(AK.paired(T[(lv, "full")], T[(lv, "tile12")])) for lv in ("s40", "s30")}
        R["T3"] = bool(R["t3"][1][0] > 0 and R["t3"][0] >= 0.10)
        R["T4"] = bool(all(e[1][0] > -0.05 and e[1][1] < 0.05 for e in R["t4"].values()))
    else:
        R["t1"] = {lv: conv(AK.paired(T[(lv, "full")], T[(lv, "k1")])) for lv in ("s40", "s30")}
        R["t2a"] = conv(AK.paired(T[("s30", "k2")], T[("s30", "k1")]))
        R["t2b"] = conv(AK.paired(T[("s30", "full")], T[("s30", "k2")]))
        R["near"] = {lv: conv(AK.paired(T[(lv, "full_nearest")], T[(lv, "full")])) for lv in ("s40", "s30")}
        R["T1"] = bool(all(e[1][0] > 0 for e in R["t1"].values()) and R["t1"]["s30"][0] >= 0.10)
        R["T2"] = bool(R["t2a"][1][0] > 0 and R["t2b"][1][0] > 0)
    return R


# ---------- figures ----------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_attr(A):
    lvs = ["s100", "s60", "s50", "s40", "s30"]
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75), sharey=True)
    for ax, tag in zip(axes, ("qwen25vl7b", "qwen3vl8b")):
        for i, (nm, c) in enumerate((("Perception", "#2a78d6"), ("Computation", "#eb6834"), ("Scale", "#1baf7a"))):
            y = [100 * A[tag]["lv"][k]["phi"][i] for k in lvs]
            ax.plot(lvs, y, color=c, linewidth=1.5, marker="o", markersize=3, label=nm)
            ax.annotate(nm[0], (4, y[-1]), xytext=(4, -2), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(TAGS[tag][1], fontsize=8, color=INK)
        ax.set_ylim(0, 48)
        style(ax)
    axes[0].set_ylabel("Shapley value (EM points)", fontsize=7, color=MUTED)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.subplots_adjust(bottom=0.27, top=0.88, left=0.15, right=0.95, wspace=0.12)
    fig.savefig(PAPER / "fig" / "attr.pdf")
    plt.close(fig)


def fig_token(K):
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75), sharey=True)
    for ax, lv in zip(axes, ("s40", "s30")):
        for tag in ("qwen25vl7b", "qwen3vl8b"):
            if not K.get(tag):
                continue
            cs = ("k1", "k1.5", "k2", "full")
            xs = [K[tag]["size"][f"{lv}|{c}"] for c in cs]
            ys = [100 * K[tag]["T"][f"{lv}|{c}"] for c in cs]
            ax.plot(xs, ys, color=COLOR[tag], linewidth=1.5, marker="o", markersize=3, label=TAGS[tag][1])
        ax.set_title(f"source {lv}", fontsize=8, color=INK)
        ax.set_xlabel("image tokens", fontsize=7, color=MUTED)
        ax.set_ylim(0, 100)
        style(ax)
    axes[0].set_ylabel("cells read correctly (%)", fontsize=7, color=MUTED)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=7, frameon=False)
    fig.subplots_adjust(bottom=0.2, top=0.78, left=0.15, right=0.97, wspace=0.12)
    fig.savefig(PAPER / "fig" / "token.pdf")
    plt.close(fig)


def fig_ladder(D):
    steps = ["Real", "Base", "G", "LG"]
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9), sharey=True)
    for ax, lv in zip(axes, ("s40", "s30")):
        for tag in TAGS:
            x = D[tag]["lv"][lv]
            ax.plot(steps, [100 * x[k] for k in ("orig", "base", "G", "LG")], color=COLOR[tag], linewidth=1.5, marker="o", markersize=3, label=TAGS[tag][1])
        ax.set_title(f"source {lv}", fontsize=8, color=INK)
        ax.set_ylim(0, 100)
        style(ax)
    axes[0].set_ylabel("cells read correctly (%)", fontsize=7, color=MUTED)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6.5, frameon=False, columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(bottom=0.12, top=0.78, left=0.15, right=0.97, wspace=0.12)
    fig.savefig(PAPER / "fig" / "ladder.pdf")
    plt.close(fig)


def ci_short(e):
    return f"${100 * e[0]:+.1f}$ {{\\scriptsize[{100 * e[1][0]:+.1f}, {100 * e[1][1]:+.1f}]}}"


def main():
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "fig").mkdir(exist_ok=True)
    A = {t: attr(t) for t in ("qwen25vl7b", "qwen3vl8b")}
    D = {t: decide(t) for t in TAGS}
    K = {t: token(t) for t in TAGS}
    for t, R in A.items():
        key = TAGS[t][0]
        put(f"{key}.attr.n", R["n"])
        put(f"{key}.attr.dshare", pm(R["dshare"][0], R["dshare"][1]))
        put(f"{key}.attr.crit", pm(R["crit3"][0], R["crit3"][1]))
        for lv in ("s100", "s30"):
            put(f"{key}.attr.{lv}.share", f"{100 * R['lv'][lv]['share']:.0f}%")
    for t, R in D.items():
        macros_decide(TAGS[t][0], R)
    for t, R in K.items():
        if not R:
            continue
        key = TAGS[t][0]
        put(f"{key}.tok.n", R["n"])
        for k, v in R["T"].items():
            put(f"{key}.tok.{k.replace('|', '.')}", f"{100 * v:.1f}")
        for k, v in R["size"].items():
            put(f"{key}.size.{k.replace('|', '.')}", f"{v:.0f}")
        for nm in ("t3", "t2a", "t2b"):
            if nm in R:
                put(f"{key}.tok.{nm}", pm(*R[nm]))
        for nm in ("t1", "t4", "near"):
            for lv, e in R.get(nm, {}).items():
                put(f"{key}.tok.{nm}.{lv}", pm(*e))
    (PAPER / "numbers.tex").write_text("".join(f"\\expandafter\\def\\csname v@{k}\\endcsname{{{v}}}\n" for k, v in sorted(V.items())))

    # Table: attribution
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Model & Level & EM & $\phi_P$ & $\phi_C+\phi_S$ & $P$ share \\", r"\midrule"]
    for t, R in A.items():
        for lv in ("s100", "s50", "s40", "s30", "photo"):
            x = R["lv"][lv]
            nm = TAGS[t][1] if lv == "s100" else ""
            L.append(f"{nm} & {lv} & {100 * x['em']:.1f} & {100 * x['phi'][0]:.1f} & {100 * (x['phi'][1] + x['phi'][2]):.1f} & {100 * x['share']:.0f}\\% \\\\")
        L.append(r"\addlinespace")
    write_table("attr", L[:-1] + [r"\bottomrule", r"\end{tabular}"])

    # Table: verdicts
    word = lambda b: "yes" if b else "no"  # noqa: E731
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Criterion & 7B & Qwen3 & InternVL \\", r"\midrule"]
    L.append("Valid & " + " & ".join(word(D[t]["valid"]["ok"]) for t in TAGS) + r" \\")
    L.append("D1 separable & " + " & ".join(",".join(D[t]["D1"]) or "no" for t in TAGS) + r" \\")
    L.append("D2 context pull & " + " & ".join(D[t]["D2"] for t in TAGS) + r" \\")
    L.append("D3 task dependence & " + " & ".join(D[t]["D3"] for t in TAGS) + r" \\")
    L.append("D4a hidden signal & " + " & ".join(word(D[t]["D4a"]) for t in TAGS) + r" \\")
    L.append("D4b verbal refusal & " + " & ".join(word(D[t]["D4b"]) for t in TAGS) + r" \\")
    L.append("D4c signal--speech gap & " + " & ".join(word(D[t]["D4c"]) for t in TAGS) + r" \\")
    if all(K.values()):
        L.append("T1 upscaling rescues & " + word(K["qwen25vl7b"]["T1"]) + " & " + word(K["qwen3vl8b"]["T1"]) + r" & -- \\")
        L.append("T2 dose response & " + word(K["qwen25vl7b"]["T2"]) + " & " + word(K["qwen3vl8b"]["T2"]) + r" & -- \\")
        L.append(r"T3 tiles matter & -- & -- & " + word(K["internvl3"]["T3"]) + r" \\")
        L.append(r"T4 upscaling inert & -- & -- & " + word(K["internvl3"]["T4"]) + r" \\")
    write_table("verdicts", L + [r"\bottomrule", r"\end{tabular}"])

    # Table: 2x2
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule", r"Model & Lvl & Real & Base & $L$ & $G$ & $LG$ & $\phi_L$ / $\phi_G$ \\", r"\midrule"]
    for t, R in D.items():
        for lv in ("s50", "s40", "s30"):
            x = R["lv"][lv]
            nm = {"qwen25vl7b": "7B", "qwen3vl8b": "Qwen3", "internvl3": "Intern"}[t] if lv == "s50" else ""
            mark = lambda e: "*" if e[1][0] > 0 or e[1][1] < 0 else ""  # noqa: E731
            L.append(f"{nm} & {lv} & {100 * x['orig']:.1f} & {100 * x['base']:.1f} & {100 * x['L']:.1f} & {100 * x['G']:.1f} & {100 * x['LG']:.1f} & "
                     f"{100 * x['phiL'][0]:+.1f}{mark(x['phiL'])} / {100 * x['phiG'][0]:+.1f}{mark(x['phiG'])} \\\\")
    write_table("factorial", L + [r"\bottomrule", r"\end{tabular}"])

    # Table: metacognition
    L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule", r"Model & Lvl & Correct & AUROC$_{lp}$ & AUROC$_{conf}$ & Refusal & Refusal gap \\", r"\midrule"]
    for t, R in D.items():
        for lv in ("s40", "s30"):
            x = R["lv"][lv]
            nm = {"qwen25vl7b": "7B", "qwen3vl8b": "Qwen3", "internvl3": "Intern"}[t] if lv == "s40" else ""
            L.append(f"{nm} & {lv} & {100 * x['orig']:.1f} & {x['au']:.2f} & {x['conf']['au_c']:.2f} & {100 * x['ref']['copyu2']['rate']:.1f} & {100 * x['ref']['copyu2']['gap'][0]:+.1f} \\\\")
    write_table("meta", L + [r"\bottomrule", r"\end{tabular}"])

    # Table: counterfactual
    L = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"Model & Lvl & $T$(cf) & $O-A$ & $T$(copy)$-T$(expr) \\", r"\midrule"]
    for t, R in D.items():
        for lv in ("s50", "s40", "s30"):
            x = R["lv"][lv]
            nm = {"qwen25vl7b": "7B", "qwen3vl8b": "Qwen3", "internvl3": "Intern"}[t] if lv == "s50" else ""
            L.append(f"{nm} & {lv} & {100 * x['cf']:.1f} & {100 * x['pull'][0]:+.1f} & {ci_short((x['task'][0], x['task'][1]))} \\\\")
    write_table("counterfactual", L + [r"\bottomrule", r"\end{tabular}"])

    fig_attr(A)
    fig_ladder(D)
    if K["qwen25vl7b"] and K["qwen3vl8b"]:
        fig_token(K)

    def plain(o):
        if isinstance(o, dict):
            return {str(k): plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, np.ndarray)):
            return [plain(v) for v in o]
        return o.item() if isinstance(o, np.generic) else o
    (ROOT / "results" / "paper_numbers.json").write_text(json.dumps(plain(dict(attr=A, decide=D, token=K)), indent=1))
    for t in TAGS:
        R = D[t]
        print(t, "n", R["n"], "valid", R["valid"]["ok"], "D1", R["D1"], "D2", R["D2"], "D3", R["D3"], "D4a", R["D4a"], "D4b", R["D4b"], "D4c", R["D4c"])
    print("crit3", {t: pm(*A[t]["crit3"]) for t in A})
    print("macros", len(V))


if __name__ == "__main__":
    main()
