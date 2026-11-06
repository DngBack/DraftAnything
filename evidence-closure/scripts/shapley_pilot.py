"""Pilot (E1 dev, exploratory): interventional error attribution with Shapley values under official EM.

Players: P = gold facts given (perception), C = arithmetic executed by Python (computation), S = gold scale (unit).
v(coalition) = EM of the pipeline with those stages repaired. Uses E8 outputs only (no new VLM calls).
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.official_metric import exact_match  # noqa: E402
from official_rescore import direct_pred  # noqa: E402
from run_e8 import safe_eval  # noqa: E402
from run_e9 import canon_scale  # noqa: E402


def has_expr(t):
    return t is not None and safe_eval(t) is not None and any(o in t for o in "+-*/")


def ev(r, P, C, S):
    txt = r["hint"] if P else r["direct"]
    expr = r["hintprog"] if P else r["program"]
    sc = r["scale"] if S else canon_scale(txt)
    if C and r["answer_type"] == "arithmetic" and has_expr(expr):  # no computation to repair elsewhere
        return exact_match([str(float(safe_eval(expr)))], sc, r)
    return exact_match(direct_pred(txt, sc), sc, r)


def shap(v, n=3):
    out = np.zeros(n)
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for k in range(n):
            for T in itertools.combinations(rest, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                a = [j in T for j in range(n)]
                b = list(a)
                b[i] = True
                out[i] += w * (v[tuple(b)] - v[tuple(a)])
    return out


def per_q(rows):
    V = {c: np.array([ev(r, *c) for r in rows]) for c in itertools.product([False, True], repeat=3)}
    return V


for tag in ("qwen25vl7b", "qwen3vl8b"):
    R = [json.loads(l) for l in open(ROOT / "results" / f"e8_{tag}.jsonl")]
    V = per_q(R)
    m = {c: x.mean() for c, x in V.items()}
    s = shap(m)
    docs = np.array([r["uid"] for r in R])
    ud = sorted(set(docs))
    idx = {d: np.where(docs == d)[0] for d in ud}
    rng = np.random.default_rng(0)
    B = []
    for _ in range(1000):
        ii = np.concatenate([idx[ud[k]] for k in rng.choice(len(ud), len(ud))])
        B.append(shap({c: x[ii].mean() for c, x in V.items()}))
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    print(f"\n{tag} n={len(R)} baseline EM {100 * m[(False, False, False)]:.1f}, all repaired {100 * m[(True, True, True)]:.1f}")
    for nm, x, a, b in zip(("perception(P)", "computation(C)", "scale(S)"), s, lo, hi):
        print(f"  {nm:15s} {100 * x:+.1f} [{100 * a:+.1f},{100 * b:+.1f}]")
    print("  unexplained residual", round(100 * (1 - m[(True, True, True)]), 1))
    print("  by type:")
    for t in ("arithmetic", "span", "multi-span", "count"):
        ii = [i for i, r in enumerate(R) if r["answer_type"] == t]
        sv = shap({c: x[ii].mean() for c, x in V.items()})
        print(f"   {t:10s} n={len(ii):3d} base {100 * V[(False, False, False)][ii].mean():5.1f} P {100 * sv[0]:+5.1f} C {100 * sv[1]:+5.1f} S {100 * sv[2]:+5.1f} full {100 * V[(True, True, True)][ii].mean():5.1f}")
