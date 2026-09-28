"""E0 audit + oracle ceiling and E1/H2 selector comparison on frozen-extractor outputs.
Selectors fit on train, thresholds on validation, reported on test. Usage: python analyze.py [MODEL_DIR]"""
import glob
import json
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import EXCLUDED, FIELDS, apply_threshold, aurc, coverage_at_risk, label, load_cord, norm_value, parse_num, threshold_at_risk

MD = sys.argv[1] if len(sys.argv) > 1 else "outputs/Qwen3-VL-4B-Instruct"
TARGETS = [0.10, 0.05, 0.02]
FL = list(FIELDS)  # all fields are scored for key-side competition
PRIMARY = [f for f in FL if f != "qty"]  # decided before test: qty is counted, not read; small ints collide
FLOOR = -30.0  # log-score assigned when the prediction is not among page candidates (UNCOVERED)


def logsoftmax(x):
    x = np.asarray(x, float)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def rows(split):
    recs = {}
    for fn in glob.glob(f"{MD}/{split}.jsonl") + glob.glob(f"{MD}/{split}.shard*.jsonl"):
        for l in open(fn):
            r = json.loads(l)
            recs[r["id"]] = r
    out = []
    for d in load_cord(split):
        r = recs.get(d["id"])
        if r is None:
            continue
        vals = r["cands"]
        # per field: log-distribution over [page candidates..., null]  (value-side competition)
        LS = {f: logsoftmax(r["scores"][f]["cands"] + [r["scores"][f]["null"]]) for f in FL}
        for f in FL:
            g = d["gold"][f]
            if g == EXCLUDED:
                continue
            p = r["fields"][f]
            pred, lps = p["pred"], np.asarray(p["lps"] or [0.0])
            ok, et = label(pred, g, d["lines"], f)
            pv = norm_value(pred)
            idx = len(vals) if pv is None else (vals.index(pv) if pv in vals else None)
            ls = LS[f]
            if idx is None:
                bind_lp = margin = key_lp = FLOOR
                argmax = 0
            else:
                bind_lp = ls[idx]
                margin = max(ls[idx] - np.delete(ls, idx).max(), FLOOR)
                argmax = int(ls.argmax() == idx)
                # key-side competition: which field claims this value most? (null: not meaningful -> 0)
                key_lp = logsoftmax([LS[h][idx] for h in FL])[FL.index(f)] if pv is not None else 0.0
            out.append({
                "doc": d["id"], "field": f, "pred": pred, "gold": g, "err": int(not ok), "etype": et,
                "is_null": int(pv is None), "missing_key": int(p.get("missing_key", False)),
                "lp_mean": lps.mean(), "lp_min": lps.min(), "lp_first": lps[0], "n_tok": len(p["lps"]),
                "on_page": int(pv is not None and idx is not None),
                "n_occ": sum(pv == parse_num(t) for t in r["surfaces"]) if pv is not None else 0,
                "n_cands": len(vals),
                "bind_lp": bind_lp, "bind_margin": margin, "bind_argmax": argmax, "key_lp": key_lp,
                "bind_entropy": float(-(np.exp(ls) * ls).sum()), "null_lp": ls[-1],
            })
    return out


BASE = ["lp_mean", "lp_min", "lp_first", "n_tok", "is_null", "missing_key"]
FUSION = BASE + ["on_page", "n_occ", "n_cands"]
BIND = ["bind_lp", "bind_margin", "bind_argmax", "key_lp", "bind_entropy", "null_lp"]


def X(R, feats):
    return np.array([[r[k] for k in feats] + [r["field"] == f for f in PRIMARY] for r in R], float)


def fit_score(tr, feats):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    m.fit(X(tr, feats), [1 - r["err"] for r in tr])
    return lambda R: m.predict_proba(X(R, feats))[:, 1]


def push_down(s, m):
    """Oracle re-ranking: items in mask m go below all others, keeping their internal order (no giant tie)."""
    return np.where(m, s - (s.max() - s.min() + 1), s)


def aurc_ci(R, sa, sb, B=2000, seed=0):
    """Paired doc bootstrap 95% CI of AURC(a) - AURC(b)."""
    groups, e = doc_index(R), np.array([r["err"] for r in R])
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        d.append(aurc(sa[ix], e[ix]) - aurc(sb[ix], e[ix]))
    return np.percentile(d, [2.5, 97.5])


def doc_index(R):
    by = {}
    for i, r in enumerate(R):
        by.setdefault(r["doc"], []).append(i)
    return list(by.values())


def boot_diff(R, sa, sb, ta, tb, B=2000, seed=0):
    """Paired doc-level bootstrap: 95% CI of coverage(a) - coverage(b) at fixed (validation) thresholds."""
    groups, e = doc_index(R), np.array([r["err"] for r in R])
    rng = np.random.default_rng(seed)
    dc = []
    for _ in range(B):
        ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        dc.append(apply_threshold(sa[ix], e[ix], ta)[0] - apply_threshold(sb[ix], e[ix], tb)[0])
    return np.percentile(dc, [2.5, 97.5])


def main():
    qty = [r for r in rows("validation") + rows("test") if r["field"] == "qty"]
    print("secondary (qty field, val+test): err_rate", round(np.mean([r["err"] for r in qty]), 3), dict(Counter(r["etype"] for r in qty if r["err"])))
    tr, va, te = ([r for r in rows(s) if r["field"] in PRIMARY] for s in ("train", "validation", "test"))
    for name, R in [("train", tr), ("validation", va), ("test", te)]:
        print(f"{name}: docs={len(doc_index(R))} decisions={len(R)} err_rate={np.mean([r['err'] for r in R]):.3f}")

    # ---------- E0: taxonomy ----------
    allR = tr + va + te
    errs = [r for r in allR if r["err"]]
    print("\n== E0 error taxonomy (all splits, frozen extractor, primary fields) ==")
    print("all errors:", dict(Counter(r["etype"] for r in errs)), "n =", len(errs), f"({len(errs) / len(allR):.3f} of decisions)")
    hc = [r for r in errs if r["lp_mean"] > np.log(0.99)]
    print("high-confidence errors (mean token prob > 0.99):", dict(Counter(r["etype"] for r in hc)), "n =", len(hc))
    print("binding(+absent) errors by field:", dict(Counter(r["field"] for r in errs if r["etype"].startswith("binding"))))
    print("(etype, value-on-page) counts:", dict(Counter((r["etype"], r["on_page"]) for r in errs)))

    # ---------- scores ----------
    scores = {
        "lp_mean": lambda R: np.array([r["lp_mean"] for r in R]),
        "lp_min": lambda R: np.array([r["lp_min"] for r in R]),
        "bind_lp (training-free)": lambda R: np.array([r["bind_lp"] for r in R]),
        "LR extractor-only": fit_score(tr, BASE),
        "LR fusion (OCR/layout)": fit_score(tr, FUSION),
        "LR binding-only": fit_score(tr, BIND),
        "LR fusion+binding": fit_score(tr, FUSION + BIND),
    }
    e_te, e_va = np.array([r["err"] for r in te]), np.array([r["err"] for r in va])
    S_te = {k: fn(te) for k, fn in scores.items()}
    S_va = {k: fn(va) for k, fn in scores.items()}

    print("\n== E0 oracle ceiling on test (in-sample curve; errors of one type pushed to reject) ==")
    for k in ["lp_mean", "LR fusion (OCR/layout)"]:
        s = S_te[k]
        for et in ["binding*", "binding", "binding_absent", "absent_halluc", "transcription", "miss"]:
            # binding* = both binding kinds = the hypothesised failure mode
            m = np.array([str(r["etype"]).startswith("binding") if et == "binding*" else r["etype"] == et for r in te])
            gains = [coverage_at_risk(push_down(s, m), e_te, t) - coverage_at_risk(s, e_te, t) for t in TARGETS]
            print(f"{k:24s} oracle[{et:14s}] n={int(m.sum()):3d} " + " ".join(f"@{t:.2f}:+{100 * g:5.1f}pp" for t, g in zip(TARGETS, gains)))

    print("\n== E1/H2: threshold from validation, applied to test (cov/risk), plus in-sample test cov ==")
    TH = {}
    for k in scores:
        cells = []
        for t in TARGETS:
            TH[k, t] = threshold_at_risk(S_va[k], e_va, t)
            c, rk = apply_threshold(S_te[k], e_te, TH[k, t])
            cells.append(f"@{t:.2f} {c:.3f}/{rk:.3f} (oracle-thr {coverage_at_risk(S_te[k], e_te, t):.3f})")
        print(f"{k:24s} AURC={aurc(S_te[k], e_te):.4f}  " + "  ".join(cells))

    print("\n== primary contrast: LR fusion+binding vs LR fusion (paired doc bootstrap, 95% CI) ==")
    a, b = "LR fusion+binding", "LR fusion (OCR/layout)"
    for t in TARGETS:
        ca, ra = apply_threshold(S_te[a], e_te, TH[a, t])
        cb, rb = apply_threshold(S_te[b], e_te, TH[b, t])
        lo, hi = boot_diff(te, S_te[a], S_te[b], TH[a, t], TH[b, t])
        print(f"target {t:.2f}: cov {ca:.3f} vs {cb:.3f}  diff {100 * (ca - cb):+.1f}pp CI[{100 * lo:+.1f},{100 * hi:+.1f}]  realised risk {ra:.3f} vs {rb:.3f}")

    lo, hi = aurc_ci(te, S_te[a], S_te[b])
    print(f"AURC {aurc(S_te[a], e_te):.4f} vs {aurc(S_te[b], e_te):.4f}  diff CI[{lo:+.4f},{hi:+.4f}]")

    # sensitivity: drop the semantically ambiguous "subtotal not printed, model copies total" decisions
    tot = {r["doc"]: norm_value(r["gold"]) for r in te + va if r["field"] == "total"}
    amb = lambda r: r["field"] == "subtotal" and norm_value(r["gold"]) is None and norm_value(r["pred"]) == tot.get(r["doc"])
    kt, kv = np.array([not amb(r) for r in te]), np.array([not amb(r) for r in va])
    print(f"\n== sensitivity: without ambiguous subtotal=total decisions (test drops {int((~kt).sum())}) ==")
    for t in TARGETS:
        cells = []
        for k in (a, b):
            th = threshold_at_risk(S_va[k][kv], e_va[kv], t)
            cells.append("%.3f/%.3f" % apply_threshold(S_te[k][kt], e_te[kt], th))
        print(f"target {t:.2f}: fusion+binding {cells[0]}  fusion {cells[1]}")
    lo, hi = aurc_ci([r for r, k_ in zip(te, kt) if k_], S_te[a][kt], S_te[b][kt])
    print(f"AURC {aurc(S_te[a][kt], e_te[kt]):.4f} vs {aurc(S_te[b][kt], e_te[kt]):.4f}  diff CI[{lo:+.4f},{hi:+.4f}]")

    print("\n== errors auto-accepted on test at validation threshold for target 0.05 ==")
    for k in scores:
        acc = S_te[k] >= TH[k, 0.05]
        print(f"{k:24s} accepted={int(acc.sum()):4d} errors={int((acc & (e_te == 1)).sum()):3d} "
              + str(dict(Counter(r["etype"] for r, a_ in zip(te, acc) if a_ and r["err"]))))


if __name__ == "__main__":
    main()
