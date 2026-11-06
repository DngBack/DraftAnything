"""Secondary analysis (docs/protocol_attr.md): 4-player Shapley P, O, C, S on arithmetic questions with a gold formula,
at the levels collected with --op (s100, s30). Written before the full data were viewed.
P1 O1 C1 needs no model: the gold derivation is executed.
"""
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
from analyze_attr import has_expr  # noqa: E402
from evc.official_metric import exact_match  # noqa: E402
from official_rescore import direct_pred  # noqa: E402
from run_e8 import safe_eval  # noqa: E402
from run_e9 import canon_scale  # noqa: E402

COAL = list(itertools.product([False, True], repeat=4))
NAMES = ("P", "O", "C", "S")
TXT = {(False, False): "direct", (True, False): "hint", (False, True): "opdirect", (True, True): "opboth"}
EXPR = {(False, False): "program", (True, False): "hintprog", (False, True): "opprog"}


def ev(r, P, O, C, S):
    txt = r[TXT[(P, O)]]
    sc = r["scale"] if S else canon_scale(txt)
    if C:
        e = r["derivation"] if P and O else r[EXPR[(P, O)]]
        if has_expr(e):
            x = float(safe_eval(e))
            c = [x, x * 100] if S and r["scale"] == "percent" else [x]
            return max(exact_match([str(v)], sc, r) for v in c)
    return exact_match(direct_pred(txt, sc), sc, r)


def shap(m, n=4):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default="attr")
    a = ap.parse_args()
    L = [f"# Shapley 4 người chơi, {a.name} {a.tag} (câu số học có công thức gold)", "",
         "| mức | n | EM gốc | EM sửa hết | φ_P | φ_O | φ_C | φ_S |", "|---|---|---|---|---|---|---|---|"]
    for lv in ("s100", "s30"):
        p = ROOT / "results" / f"{a.name}_{a.tag}_{lv}.jsonl"
        if not p.exists():
            continue
        R = [json.loads(l) for l in open(p)]
        R = [r for r in R if r["answer_type"] == "arithmetic" and r.get("opboth") is not None]
        V = {c: np.array([ev(r, *c) for r in R]) for c in COAL}
        docs = np.array([r["uid"] for r in R])
        ud = sorted(set(docs))
        idx = {x: np.where(docs == x)[0] for x in ud}
        rng = np.random.default_rng(0)
        s = shap({c: V[c].mean() for c in COAL})
        B = []
        for _ in range(1000):
            ii = np.concatenate([idx[ud[k]] for k in rng.choice(len(ud), len(ud))])
            B.append(shap({c: V[c][ii].mean() for c in COAL}))
        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        cells = [f"{100 * s[i]:+.1f} [{100 * lo[i]:+.1f}, {100 * hi[i]:+.1f}]" for i in range(4)]
        L.append(f"| {lv} | {len(R)} | {100 * V[COAL[0]].mean():.1f} | {100 * V[COAL[-1]].mean():.1f} | " + " | ".join(cells) + " |")
    out = ROOT / "results" / f"{a.name}_{a.tag}_op.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
