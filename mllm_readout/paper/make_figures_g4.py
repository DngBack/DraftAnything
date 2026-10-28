"""G4 figures: label cost of repair (Fig. 1) and what steering writes (Fig. 3).
Run from mllm_readout/: PYTHONPATH=paper python paper/make_figures_g4.py"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_tables import R

INK, MUTED = "#0b0b0b", "#52514e"
# categorical slots of the validated default palette, fixed order
COL = {"ext": "#2a78d6", "mllm0": "#8e5fd3", "ans": "#eb6834", "cal": "#1baf7a", "icl": "#e0457b", "zero": "#d4a20f"}
G4 = ["qwen35_2b", "qwen35_4b", "qwen35_9b", "qwen3vl4b", "qwen3vl8b"]
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED})


def acc(f, ref=None):
    """accuracy of a g4 npz; with ref (another npz), also the reference accuracy on the same items"""
    d = np.load(f)
    a = np.mean(d["pred"] == d["y"])
    if ref is None:
        return a
    r = np.load(ref)
    m = dict(zip(r["idx"].tolist(), r["pred"].tolist()))
    return a, np.mean([m.get(int(i), -9) == y for i, y in zip(d["idx"], d["y"])])


def zero_label(name):
    """gain (points) of label-free repairs over the unedited model, mean over the models where they were run"""
    out = {}
    I = pd.read_csv(f"{R}/g4_matrix_items.csv")
    I = I[I.set == name].assign(G3=lambda z: (z.type == "CORRECT").astype(float))
    c = I.groupby("model")[["G3", "R1", "R2", "R3", "R3x", "R5"]].mean()
    for k in ("R1", "R2", "R3", "R3x", "R5"):
        out[k] = 100 * (c[k] - c.G3).mean()
    th = [acc(f"{R}/g4/{m}_{name}_thinkfix.npz", f"{R}/g4/{m}_{name}_plain.npz") for m in G4[:3]]
    out["R4"] = 100 * np.mean([a - b for a, b in th])
    if name == "illusions":
        ch = [acc(f, f.replace("_channel", "_plain")) for f in glob.glob(f"{R}/g4/*_illusions_channel.npz")]
        out["channel"] = 100 * np.mean([a - b for a, b in ch])
    return out


ICL_MODELS = ["qwen3vl4b", "qwen3vl8b", "qwen35_4b", "qwen35_9b", "gemma3_4b", "internvl35_8b"]


def icl_base(m):
    """free generation on the ICL test items: the G4 plain run, else ICL k=0 if parsed, else the G3 decision-step answer"""
    k0 = np.load(f"{R}/g4/icl_{m}_k0.npz") if glob.glob(f"{R}/g4/icl_{m}_k0.npz") else None
    if glob.glob(f"{R}/g4/{m}_illusions_plain.npz"):
        p = np.load(f"{R}/g4/{m}_illusions_plain.npz")
        return dict(zip(p["idx"].tolist(), (p["pred"] == p["y"]).tolist()))
    if k0 is not None and np.mean(k0["pred"] < 0) < 0.1:
        return dict(zip(k0["idx"].tolist(), (k0["pred"] == k0["y"]).tolist()))
    d = np.load(glob.glob(f"{R}/g3*/{m}_illusions_states.npz")[0])
    return dict(zip(d["idx"].tolist(), (d["ans"] == d["y"]).tolist()))


def icl_gains(fmt="", models=ICL_MODELS, ks=(1, 2, 4, 8)):
    """{k: [gain in points per model]} of k-shot in-context examples over free generation on the same items"""
    out = {}
    for m in models:
        base = icl_base(m)
        for k in ks:
            f = f"{R}/g4/icl_{m}{fmt}_k{k}.npz"
            if glob.glob(f):
                z = np.load(f)
                out.setdefault(k, []).append(100 * (np.mean(z["pred"] == z["y"]) - np.mean([base.get(int(i), 0) for i in z["idx"]])))
    return out


def labelcost():
    D = pd.read_csv(f"{R}/explore_labelcost.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, name, kmax in ((axes[0], "illusions", 32), (axes[1], "countbench", 16)):
        x = D[(D.set == name) & (D.k <= kmax)]
        g = x.assign(**{c: 100 * (x[c + "_ridge"] - x.raw) for c in ("ext", "mllm0", "ans", "cal")}).groupby("k")[
            ["ext", "mllm0", "ans", "cal"]].mean()
        lab = {"ext": "frozen SigLIP2", "mllm0": "MLLM image tokens (L0)", "ans": "MLLM decision state",
               "cal": "model's own logits"}
        for c in ("ext", "mllm0", "ans", "cal"):
            ax.plot(np.log2(g.index), g[c], "-o", color=COL[c], ms=2.5, lw=1.3, label=lab[c])
        z = zero_label(name)
        if name == "illusions":
            icl = icl_gains()
            ks = sorted(icl)
            ax.plot(np.log2(ks), [np.mean(icl[k]) for k in ks], "--s", color=COL["icl"], ms=3, lw=1.1,
                    label="in-context examples (model, 6 models)")
        # label-free repairs at x = "0 labels"; labels spread vertically
        zk = [("R4", "think"), ("R2", "guidance"), ("R5", "2$\\times$ image")] + ([("channel", "boxes")] if "channel" in z else [])
        zk.sort(key=lambda t: z[t[0]])
        lo = 4.0 if name == "illusions" else -1.5
        for j, (k, t) in enumerate(zk):
            ax.scatter(-1.7, z[k], marker="D", s=12, color=COL["zero"], zorder=4, edgecolor="white", linewidth=0.4)
            ax.annotate(t, (-1.7, z[k]), xytext=(-1.45, lo + j * (2.9 if name == "illusions" else 3.4)), fontsize=5.3,
                        color=MUTED, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.3))
        # label-fitted repairs placed at their label budget (labels per class per question)
        n_cls, n_lab = (2 * 12, 0.8 * 792) if name == "illusions" else (9, 0.8 * 437)
        kb = np.log2(n_lab / n_cls)
        pts = [("R3", "steering,\nown probe"), ("R3x", "steering,\nSigLIP2 probe")] if name == "illusions" else \
              [("R1", "calibrator"), ("R3", "steering,\nown probe")]
        for k, t in pts:
            ax.scatter(kb, z[k], marker="*" if k != "R1" else "P", s=40 if k != "R1" else 22, color=INK, zorder=5,
                       edgecolor="white", linewidth=0.4)
            ax.annotate(t, (kb, z[k]), xytext=(max(np.log2(kmax), kb) + 0.35, z[k] - (3 if k == "R3x" else 0)), fontsize=5.3,
                        color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.3))
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(-0.5, color=MUTED, lw=0.4, ls=":")
        ticks = [-1.7] + list(np.log2(sorted(g.index)))
        ax.set_xticks(ticks, ["0"] + [str(int(2 ** t)) for t in ticks[1:]])
        ax.set_xlim(-2.1, max(np.log2(kmax), kb) + 1.9)
        if name == "countbench":
            ax.set_ylim(-26, 12)
            ax.text(0.2, -24.5, "SigLIP2 and MLLM image tokens: $-$53 to $-$38 (off scale)", fontsize=5.3, color=MUTED)
        ax.set_xlabel("labelled items per class" + (" and wording" if name == "illusions" else ""), color=INK)
        ax.set_title({"illusions": "Edited illusions (errors inherited from the encoder)",
                      "countbench": "CountBench (count computed by the LLM)"}[name], fontsize=7, color=INK)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("gain over the model (points)", color=INK)
    axes[0].legend(loc="upper left", frameon=False, fontsize=5.5, handlelength=1.6, labelspacing=0.25, borderaxespad=0)
    fig.tight_layout(w_pad=1.0)
    fig.savefig("paper/fig/labelcost.pdf", bbox_inches="tight")
    fig.savefig("paper/fig/labelcost.png", dpi=200, bbox_inches="tight")


def disguise():
    d = pd.read_csv(f"{R}/explore_disguise.csv")
    d = d[d.target.isin(["own", "ext", "oracle", "anti", "random", "shuffle"])]
    mk = {"own": ("o", "own probe"), "ext": ("^", "SigLIP2 probe"), "oracle": ("P", "true class"),
          "anti": ("X", "wrong class"), "random": ("s", "random direction"), "shuffle": ("v", "shuffled labels")}
    sc = {"illusions": "#2a78d6", "countbench": "#eb6834"}
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.7, ls="--")
    for t, (m, lab) in mk.items():
        for s, c in sc.items():
            x = d[(d.target == t) & (d.set == s)]
            tx = x.probe_acc if t not in ("random", "shuffle") else x.raw
            ax.scatter(100 * tx, 100 * x.acc, marker=m, s=16, color=c, edgecolor="white", linewidth=0.4, zorder=3,
                       alpha=0.9 if t not in ("random", "shuffle") else 0.45)
    k = d[d.target.isin(["own", "ext", "oracle", "anti"])]
    r = np.corrcoef(k.acc, k.probe_acc)[0, 1]
    hs = [plt.Line2D([], [], marker=m, ls="", color=MUTED, ms=4, label=lab) for m, lab in mk.values()]
    hs += [plt.Line2D([], [], marker="o", ls="", color=c, ms=4, label={"illusions": "illusions", "countbench": "CountBench"}[s])
           for s, c in sc.items()]
    ax.legend(handles=hs, loc="upper left", frameon=False, fontsize=5.3, labelspacing=0.2, handletextpad=0.1, ncol=2,
              columnspacing=0.6, borderaxespad=0)
    ax.set_xlabel("accuracy of the steering target (%)\n(random / shuffled: raw model)", color=INK)
    ax.set_ylabel("accuracy after steering (%)", color=INK)
    ax.text(98, 3, f"$r$ = {r:.2f} over {len(k)} cells\n(probe and class targets)", ha="right", va="bottom", fontsize=5.8,
            color=MUTED)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig("paper/fig/disguise.pdf", bbox_inches="tight")
    fig.savefig("paper/fig/disguise.png", dpi=200, bbox_inches="tight")
    print(f"disguise r = {r:.3f} (n = {len(k)})")


if __name__ == "__main__":
    labelcost()
    disguise()
    for s in ("illusions", "countbench"):
        print(s, {k: round(v, 1) for k, v in zero_label(s).items()})
