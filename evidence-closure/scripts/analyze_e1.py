"""Aggregate E1 jsonl: accuracy per condition, paired deltas with doc-level bootstrap, budget curves."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RHOS = (0.02, 0.05, 0.1, 0.25, 0.5)
PAIRS = [("C_attn@{r}", "B_attn@{r}"), ("C_gold@{r}", "B_gold@{r}"), ("C_gold@{r}", "L_gold@{r}"),
         ("full", "C_attn@{r}")]
FIXED = [("closure", "values"), ("closure", "local"), ("full", "closure")]


def boot(recs, a, b, key="v", n=2000, seed=0):
    by_doc = defaultdict(list)
    for r in recs:
        x, y = r["res"][a][key], r["res"][b][key]
        if x is None or y is None:
            continue
        by_doc[r["uid"]].append(float(x) - float(y))
    docs = list(by_doc)
    if not docs:
        return None
    sums = np.array([sum(by_doc[d]) for d in docs])
    cnts = np.array([len(by_doc[d]) for d in docs])
    est = sums.sum() / cnts.sum()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(docs), len(docs))
        bs.append(sums[i].sum() / cnts[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, int(cnts.sum())


def acc(recs, c, key="v"):
    xs = [r["res"][c][key] for r in recs if r["res"][c][key] is not None]
    return (np.mean(xs), len(xs)) if xs else (float("nan"), 0)


def groups(recs):
    g = {"all": recs}
    g["single"] = [r for r in recs if r["n_facts"] == 1]
    g["multi"] = [r for r in recs if r["n_facts"] > 1]
    for t in ("span", "arithmetic", "multi-span"):
        g[t] = [r for r in recs if r["answer_type"] == t]
    g["has_scale"] = [r for r in recs if r["scale"]]
    return g


def fmt(x):
    if x is None:
        return "–"
    e, lo, hi, n = x
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{star}"


def report(tag):
    recs = [json.loads(l) for l in open(ROOT / "results" / f"e1_{tag}.jsonl")]
    lines = [f"# E1 — {tag}", "", f"n = {len(recs)} câu, {len({r['uid'] for r in recs})} tài liệu. "
             f"Visual token đầy đủ trung bình N = {np.mean([r['N'] for r in recs]):.0f}; "
             f"fact chiếm {np.mean([r['n_fact_cells'] for r in recs]):.1f} ô, "
             f"fact+closure {np.mean([r['n_clos_cells'] for r in recs]):.1f} ô.", ""]
    conds = ["full", "values", "local", "closure"] + [f"{m}@{r}" for r in RHOS for m in
                                                      ("B_attn", "C_attn", "B_gold", "L_gold", "C_gold")]
    lines += ["## Độ chính xác theo điều kiện", "",
              "| Điều kiện | K trung bình | value acc | scale acc (câu có scale) | fact recall | relation recall |",
              "|---|---|---|---|---|---|"]
    for c in conds:
        v, n = acc(recs, c)
        s, ns = acc(recs, c, "s")
        K = np.mean([r["res"][c]["K"] for r in recs])
        fr = np.mean([r["res"][c]["fact_rec"] for r in recs])
        rr = np.mean([r["res"][c]["rel_rec"] for r in recs])
        lines.append(f"| `{c}` | {K:.0f} | {100 * v:.1f} | {100 * s:.1f} | {100 * fr:.0f}% | {100 * rr:.0f}% |")
    lines += ["", "## Hiệu cặp (điểm %, CI 95% bootstrap theo tài liệu; * = CI không chứa 0)", ""]
    G = groups(recs)
    for key, kname in (("v", "value acc"), ("s", "scale acc")):
        lines += [f"### {kname}", "", "| Cặp | " + " | ".join(f"{g} (n={len(G[g])})" for g in G) + " |",
                  "|---|" + "---|" * len(G)]
        for a, b in PAIRS:
            for r in RHOS:
                aa, bb = a.format(r=r), b.format(r=r)
                lines.append(f"| `{aa}` − `{bb}` | " + " | ".join(fmt(boot(G[g], aa, bb, key)) for g in G) + " |")
        for a, b in FIXED:
            lines.append(f"| `{a}` − `{b}` | " + " | ".join(fmt(boot(G[g], a, b, key)) for g in G) + " |")
        lines.append("")
    (ROOT / "results" / f"e1_{tag}.md").write_text("\n".join(lines))
    summary = {c: acc(recs, c)[0] for c in conds}
    json.dump(summary, open(ROOT / "results" / f"e1_{tag}_summary.json", "w"), indent=1)
    plot(recs, tag)
    print("\n".join(lines))


def plot(recs, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    full = acc(recs, "full")[0]
    styles = {"B_attn": ("#4c78a8", "-", "B attn (độc lập)"), "C_attn": ("#e45756", "-", "C attn (closure)"),
              "B_gold": ("#4c78a8", "--", "B gold-seed"), "L_gold": ("#999999", "--", "L gold-seed (locality)"),
              "C_gold": ("#e45756", "--", "C gold-seed")}
    for ax, key, title in ((axes[0], "v", "Value accuracy"), (axes[1], "s", "Scale accuracy (câu có scale)")):
        for m, (col, ls, lab) in styles.items():
            ax.plot([100 * r for r in RHOS], [100 * acc(recs, f"{m}@{r}", key)[0] for r in RHOS],
                    color=col, ls=ls, marker="o", label=lab)
        ax.axhline(100 * acc(recs, "full", key)[0], color="black", lw=1, label="full")
        ax.set_xscale("log")
        ax.set_xticks([100 * r for r in RHOS], [f"{int(100 * r)}%" for r in RHOS])
        ax.set_xlabel("ngân sách visual token (% của full)")
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"E1 TAT-DQA dev — {tag}")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"e1_{tag}_budget.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen25vl7b")
    report(ap.parse_args().tag)
