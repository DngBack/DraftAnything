"""Supplementary analyses for the report: (1) bootstrap that also re-selects the threshold on resampled
validation docs, (2) per-field error table, (3) risk-coverage figure. Usage: python supplement.py"""
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze as A
from common import apply_threshold, norm_value, rc_curve, threshold_at_risk

MODELS = {"Qwen3-VL-4B": "outputs/Qwen3-VL-4B-Instruct", "Qwen2-VL-2B": "outputs/Qwen2-VL-2B-Instruct"}
TARGETS = [0.10, 0.05, 0.02]
NAMES = {"fb": "LR fusion+binding", "f": "LR fusion (OCR/layout)"}


def resample(groups, rng):
    return np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])


def full_boot(va, te, Sva, Ste, B=1000, seed=0):
    """Resample validation AND test docs; re-pick thresholds on the validation resample each time."""
    ev, et = np.array([r["err"] for r in va]), np.array([r["err"] for r in te])
    gv, gt = A.doc_index(va), A.doc_index(te)
    rng = np.random.default_rng(seed)
    out = {t: {"diff": [], "risk_fb": [], "risk_f": []} for t in TARGETS}
    for _ in range(B):
        iv, it = resample(gv, rng), resample(gt, rng)
        for t in TARGETS:
            c, r = {}, {}
            for k in ("fb", "f"):
                th = threshold_at_risk(Sva[k][iv], ev[iv], t)
                c[k], r[k] = apply_threshold(Ste[k][it], et[it], th)
            out[t]["diff"].append(c["fb"] - c["f"])
            out[t]["risk_fb"].append(r["fb"])
            out[t]["risk_f"].append(r["f"])
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (mname, md) in zip(axes, MODELS.items()):
        A.MD = md
        tr, va, te = ([r for r in A.rows(s) if r["field"] in A.PRIMARY] for s in ("train", "validation", "test"))
        fits = {"fb": A.fit_score(tr, A.FUSION + A.BIND), "f": A.fit_score(tr, A.FUSION)}
        tot = {r["doc"]: norm_value(r["gold"]) for r in va + te if r["field"] == "total"}
        amb = lambda r: r["field"] == "subtotal" and norm_value(r["gold"]) is None and norm_value(r["pred"]) == tot.get(r["doc"])

        for tag, keep in [("all decisions", lambda r: True), ("without ambiguous subtotal=total", lambda r: not amb(r))]:
            v, t_ = [r for r in va if keep(r)], [r for r in te if keep(r)]
            Sva = {k: f(v) for k, f in fits.items()}
            Ste = {k: f(t_) for k, f in fits.items()}
            bs = full_boot(v, t_, Sva, Ste)
            print(f"\n## {mname} - {tag}: full bootstrap (val+test resampled, threshold re-selected), B=1000")
            for t in TARGETS:
                d = np.array(bs[t]["diff"])
                rf = np.array(bs[t]["risk_fb"], float)
                rb = np.array(bs[t]["risk_f"], float)
                print(f"target {t:.2f}: cov diff median {100 * np.median(d):+.1f}pp  95% CI [{100 * np.percentile(d, 2.5):+.1f}, {100 * np.percentile(d, 97.5):+.1f}]"
                      f"  P(diff>0)={np.mean(d > 0):.2f}  P(risk>target): fusion+binding {np.nanmean(rf > t):.2f}, fusion {np.nanmean(rb > t):.2f}")

        allR = tr + va + te
        print(f"\n## {mname} - per-field decisions / errors (all splits, primary fields)")
        for f in A.PRIMARY:
            R = [r for r in allR if r["field"] == f]
            c = Counter(r["etype"] for r in R if r["err"])
            print(f"{f:9s} n={len(R):4d} gold-present={sum(norm_value(r['gold']) is not None for r in R):4d} err={sum(c.values()):4d} "
                  f"binding={c['binding']:3d} binding_absent={c['binding_absent']:3d} miss={c['miss']:3d} transcription={c['transcription']:3d} absent_halluc={c['absent_halluc']:3d}")

        e = np.array([r["err"] for r in te])
        curves = {"mean token logprob": np.array([r["lp_mean"] for r in te]), "LR fusion (OCR/layout)": fits["f"](te),
                  "LR fusion+binding": fits["fb"](te)}
        for lab, s in curves.items():
            cov, risk, _ = rc_curve(s, e)
            ax.step(cov, risk, where="post", label=lab)
        for t in (0.05, 0.02):
            ax.axhline(t, color="grey", lw=0.6, ls="--")
        ax.set(title=f"{mname} - CORD test (100 docs)", xlabel="coverage", ylabel="risk (error rate of accepted)", ylim=(0, 0.25))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/risk_coverage_test.png", dpi=150)
    print("\nsaved results/risk_coverage_test.png")


if __name__ == "__main__":
    main()
