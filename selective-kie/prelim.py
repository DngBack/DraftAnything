"""Preliminary result while GPUs are blocked: pool all processed docs, out-of-fold selector scores via 5-fold
GroupKFold by document, risk-coverage on the pooled OOF scores. Not the pre-registered split protocol."""
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupKFold

import analyze as A
from common import aurc, coverage_at_risk

R = [r for s in ("train", "validation", "test") for r in A.rows(s) if r["field"] in A.PRIMARY]
docs = np.array([r["doc"] for r in R])
e = np.array([r["err"] for r in R])
print(f"docs={len(set(docs))} decisions={len(R)} err_rate={e.mean():.3f}")
errs = [r for r in R if r["err"]]
print("errors:", dict(Counter(r["etype"] for r in errs)))
print("high-conf errors (p>0.99):", dict(Counter(r["etype"] for r in errs if r["lp_mean"] > np.log(0.99))))

FEATS = {"LR extractor-only": A.BASE, "LR fusion (OCR/layout)": A.FUSION, "LR binding-only": A.BIND,
         "LR fusion+binding": A.FUSION + A.BIND}
S = {"lp_mean": np.array([r["lp_mean"] for r in R]), "bind_lp (training-free)": np.array([r["bind_lp"] for r in R])}
for k, feats in FEATS.items():
    S[k] = np.zeros(len(R))
    for tr, te in GroupKFold(5).split(R, groups=docs):
        S[k][te] = A.fit_score([R[i] for i in tr], feats)([R[i] for i in te])

TARGETS = [0.05, 0.02, 0.01]
bind = np.array([str(r["etype"]).startswith("binding") for r in R])
print("\nscore                     AURC     " + "  ".join(f"cov@{t:.2f}" for t in TARGETS) + "   | oracle(binding*) cov")
for k, s in S.items():
    orc = A.push_down(s, bind)
    print(f"{k:24s} {aurc(s, e):.4f}  " + "  ".join(f"{coverage_at_risk(s, e, t):8.3f}" for t in TARGETS)
          + "   | " + "  ".join(f"{coverage_at_risk(orc, e, t):.3f}" for t in TARGETS))

# paired doc bootstrap of in-sample coverage@risk difference
groups = A.doc_index(R)
rng = np.random.default_rng(0)
a, b = S["LR fusion+binding"], S["LR fusion (OCR/layout)"]
D = {t: [] for t in TARGETS}
dA = []
for _ in range(1000):
    ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
    for t in TARGETS:
        D[t].append(coverage_at_risk(a[ix], e[ix], t) - coverage_at_risk(b[ix], e[ix], t))
    dA.append(aurc(a[ix], e[ix]) - aurc(b[ix], e[ix]))
print("\nfusion+binding minus fusion (95% doc-bootstrap CI):")
for t in TARGETS:
    print(f"  cov@{t:.2f}: {100 * (coverage_at_risk(a, e, t) - coverage_at_risk(b, e, t)):+.1f}pp  CI [{100 * np.percentile(D[t], 2.5):+.1f}, {100 * np.percentile(D[t], 97.5):+.1f}]")
print(f"  AURC: {aurc(a, e) - aurc(b, e):+.4f}  CI [{np.percentile(dA, 2.5):+.4f}, {np.percentile(dA, 97.5):+.4f}]")
