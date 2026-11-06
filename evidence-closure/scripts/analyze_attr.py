"""Shapley attribution of EM loss per image-quality level (players P, C, S; O reported where collected)."""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.degrade import LEVELS  # noqa: E402
from evc.official_metric import exact_match  # noqa: E402
from official_rescore import direct_pred  # noqa: E402
from run_e8 import read_recall, safe_eval  # noqa: E402
from run_e9 import canon_scale  # noqa: E402


def has_expr(t):
    return t is not None and safe_eval(t) is not None and any(o in t for o in "+-*/")


def ev(r, P, C, S):
    """EM with the stages in the coalition repaired. S = gold scale, including the x100 percent convention."""
    txt = r["hint"] if P else r["direct"]
    sc = r["scale"] if S else canon_scale(txt)
    if C and r["answer_type"] == "arithmetic":
        e = r["hintprog"] if P else r["program"]
        if has_expr(e):
            x = float(safe_eval(e))
            c = [x, x * 100] if S and r["scale"] == "percent" else [x]
            return max(exact_match([str(v)], sc, r) for v in c)
    return exact_match(direct_pred(txt, sc), sc, r)


COAL = list(itertools.product([False, True], repeat=3))


def shap(m, n=3):
    out = np.zeros(n)
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for k in range(n):
            for T in itertools.combinations(rest, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                a = tuple(j in T for j in range(n))
                b = tuple(j in T or j == i for j in range(n))
                out[i] += w * (m[b] - m[a])
    return out


def stats(V, ii):
    m = {c: V[c][ii].mean() for c in COAL}
    s = shap(m)
    gain = s.sum()
    return s, s / gain if gain > 0 else s * 0, m[COAL[0]], m[COAL[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default="attr")
    a = ap.parse_args()
    data = {}
    for lv in LEVELS:
        p = ROOT / "results" / f"{a.name}_{a.tag}_{lv}.jsonl"
        if p.exists():
            data[lv] = {json.loads(l)["qid"]: json.loads(l) for l in open(p)}
    common = sorted(set.intersection(*[set(d) for d in data.values()]))
    docs = np.array([data[next(iter(data))][q]["uid"] for q in common])
    V = {lv: {c: np.array([ev(d[q], *c) for q in common]) for c in COAL} for lv, d in data.items()}
    ud = sorted(set(docs))
    idx = {x: np.where(docs == x)[0] for x in ud}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[ud[k]] for k in rng.choice(len(ud), len(ud))]) for _ in range(1000)]
    all_i = np.arange(len(common))
    L = [f"# Attribution {a.name} {a.tag}, n = {len(common)} câu chung, {len(ud)} tài liệu", "",
         "| mức | EM gốc | EM sửa hết | φ_P | φ_C | φ_S | tỷ phần P | đọc đủ (program, số học) |", "|---|---|---|---|---|---|---|---|"]
    B = {lv: [stats(V[lv], b) for b in boots] for lv in V}
    for lv, d in data.items():
        s, sh, b0, b1 = stats(V[lv], all_i)
        cis = np.percentile([x[0] for x in B[lv]], [2.5, 97.5], axis=0)
        cell = [f"{100 * s[i]:+.1f} [{100 * cis[0][i]:+.1f}, {100 * cis[1][i]:+.1f}]" for i in range(3)]
        rd = [read_recall(d[q]["program"], d[q]["facts"]) >= 1 for q in common if d[q]["answer_type"] == "arithmetic"]
        L.append(f"| {lv} | {100 * b0:.1f} | {100 * b1:.1f} | " + " | ".join(cell) + f" | {100 * sh[0]:.0f} % | {100 * np.mean(rd):.1f} |")
    if "s100" in V:
        L += ["", "Thay đổi so với s100 (cùng bootstrap theo tài liệu):", ""]
        for lv in V:
            if lv == "s100":
                continue
            dabs = [x[0][0] - y[0][0] for x, y in zip(B[lv], B["s100"])]
            dsh = [x[1][0] - y[1][0] for x, y in zip(B[lv], B["s100"])]
            s1, sh1, *_ = stats(V[lv], all_i)
            s0, sh0, *_ = stats(V["s100"], all_i)
            la, ha = np.percentile(dabs, [2.5, 97.5])
            ls, hs = np.percentile(dsh, [2.5, 97.5])
            L.append(f"- {lv}: Δφ_P = {100 * (s1[0] - s0[0]):+.1f} [{100 * la:+.1f}, {100 * ha:+.1f}], "
                     f"Δ tỷ phần P = {100 * (sh1[0] - sh0[0]):+.1f} [{100 * ls:+.1f}, {100 * hs:+.1f}]")
    out = ROOT / "results" / f"{a.name}_{a.tag}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
