#!/usr/bin/env python3
"""One summary of every development result on disk -> results/SUMMARY.md (re-runnable; numbers for the paper).

PHOEBI: pooled 5 dev folds, unit = combination. BBBC041 / PanNuke: out-of-fold images, unit = image.
For instance datasets also: tile localisation AUROC (methods saved with tile evidence) and context
reliance at fixed own evidence (resolved coefficients; mean |context| / logit sd).

    python scripts/summarize.py [--n_boot 300]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev import metrics  # noqa: E402
from popev.experiment import load_preds  # noqa: E402

DEV = ROOT / "results/dev"
spec = importlib.util.spec_from_file_location("ai", ROOT / "scripts/analyze_instances.py")
ai = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai)

PHOEBI_K = ["bs", "bt", "fj", "ka", "mx", "pf"]
PHOEBI_RUNS = ["probe_cls", "probe_patchmean", "popevidence_cls", "popevidence_tile_cls", "cla_phoebi",
               "mix_phoebi", "tileonly_phoebi"]
INST = {"bbbc041": ["bbbc041", "bbbc041_tile", "bbbc041_m1", "bbbc041_abmil", "cla_bbbc041", "bbbc041_nobce"],
        "pannuke": ["pannuke", "pannuke_abmil", "cla_pannuke", "pannuke_nobce"]}


def runs(dirs, n):
    seen = {}
    for d in dirs:
        for f in sorted((DEV / d).glob(f"preds_*_n{n}.npz")):
            name = f.name[len("preds_"):-len(f"_n{n}.npz")]
            key = name
            if d == "probe_patchmean":
                key = f"{name} (patchmean)"
            seen.setdefault(key, (d, name))
    return seen


def phoebi(n_boot):
    L = ["## PHOEBI — pooled development folds (26 combinations; unit = combination)", ""]
    for n in (1, 4):
        R = runs(PHOEBI_RUNS, n)
        if "mean_lin" not in R:
            continue
        ref = load_preds(DEV / R["mean_lin"][0], "mean_lin", n)
        L += [f"### n_fields = {n}", "", "| method | macro AP | mean AUROC | Δ AUROC vs mean_lin | own>0 resolved | " +
              " | ".join(PHOEBI_K) + " |", "|---|---|---|---|---|" + "---|" * len(PHOEBI_K)]
        rows = []
        for key, (d, name) in R.items():
            S, P, Y, C, _ = load_preds(DEV / d, name, n)
            if len(Y) != len(ref[2]) or (Y != ref[2]).any():
                continue
            r = metrics.report(S, Y, P, C, PHOEBI_K, n_boot=n_boot)
            dd = metrics.paired_delta((S, P), ref[:2], Y, C, n_boot=n_boot) if key != "mean_lin" else None
            at = metrics.context_attribution(S.mean(0), Y, C, n_boot=200)
            rows.append((r["auroc_mean"]["point"], f"| {key} | {metrics.fmt(r['macro_ap'])} | {metrics.fmt(r['auroc_mean'])} | "
                         + (metrics.fmt(dd["auroc_mean"]) if dd else "—") + f" | {at['positive_resolved']} | "
                         + " | ".join(f"{r['per_species'][k]['auroc']:.2f}" for k in PHOEBI_K) + " |"))
        L += [r for _, r in sorted(rows, key=lambda t: -t[0])] + [""]
    L += ["Reference: ProtoMatch, pooled 27 test combinations (PHOEBI Table 19): mean AUROC 0.63 [0.55, 0.71].", ""]
    return L


def context_share(S, Y, rows, IC, nb, seed=0):
    Sm = S.mean(0).clip(1e-4, 1 - 1e-4); Lg = np.log(Sm / (1 - Sm)); cnt = IC[rows].astype(float); K = Y.shape[1]
    def mag(ii):
        return np.mean([np.abs(ai.fit(Lg[ii], Y[ii], cnt[ii], k)[3:]).mean() for k in range(K)]) / Lg[ii].std(0).mean()
    rng = np.random.default_rng(seed)
    bs = [mag(rng.integers(0, len(Y), len(Y))) for _ in range(nb)]
    return mag(np.arange(len(Y))), np.percentile(bs, [2.5, 97.5])


def instance(ds, n_boot):
    ai.DS, ai.K = ds, ai.CLASSES[ds]; K = ai.K
    TL = np.load(ROOT / f"cache/tile_labels_{ds}.npy"); IC = np.load(ROOT / f"cache/instance_counts_{ds}.npy")
    R = runs(INST[ds], 1)
    if "mean_lin" not in R:
        return []
    ref = load_preds(DEV / R["mean_lin"][0], "mean_lin", 1)
    fallback = None
    L = [f"## {ds} — out-of-fold images (unit = image)", "",
         "| method | macro AP | mean AUROC | decorrelated AUROC | Δ AUROC vs mean_lin | context share [95 % CI] | " +
         " | ".join(f"loc {k[:4]}" for k in K) + " |", "|---|---|---|---|---|---|" + "---|" * len(K)]
    rows = []
    for key, (d, name) in R.items():
        z = np.load(DEV / d / f"preds_{name}_n1.npz")
        S, P, Y, C = z["S"], z["P"], z["Y"].astype(float), z["C"]
        if "R" in z.files:
            rr = z["R"][:, 0]
        else:
            fallback = fallback if fallback is not None else ai.recompute_rows(); rr = fallback
        r = metrics.report(S, Y, P, C, K, n_boot=n_boot)
        dd = metrics.paired_delta((S, P), ref[:2], Y, C, n_boot=n_boot) if key != "mean_lin" else None
        cs, ci = context_share(S, Y, rr, IC, max(n_boot // 3, 100))
        loc = ["—"] * len(K)
        if "A" in z.files:
            A = z["A"].astype(np.float32).mean(0); G = TL[rr]
            loc = [f"{metrics.auroc(A[..., k].ravel(), G[..., k].ravel()):.2f}" for k in range(len(K))]
        dec = metrics.decorrelated(S, Y, P)["auroc_mean"]
        rows.append((r["auroc_mean"]["point"], f"| {key} | {metrics.fmt(r['macro_ap'])} | {metrics.fmt(r['auroc_mean'])} | {dec:.3f} | "
                     + (metrics.fmt(dd["auroc_mean"]) if dd else "—") + f" | {cs:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | " + " | ".join(loc) + " |"))
    L += [r for _, r in sorted(rows, key=lambda t: -t[0])]
    L += ["", "`loc k` = AUROC of the method's per-tile evidence for k against instance-centre-in-tile ground truth "
          "(0.5 = chance, < 0.5 = anti-localised). `context share` = mean |context coefficient| at fixed own presence "
          "and count, divided by the logit sd.", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n_boot", type=int, default=300); a = ap.parse_args()
    L = ["# PopEvidence — development results summary", "",
         "Generated by `scripts/summarize.py` from saved predictions. Development only; all test sets unopened. "
         f"Bootstrap resamples: {a.n_boot}.", ""]
    L += phoebi(a.n_boot)
    for ds in INST:
        L += instance(ds, a.n_boot)
    out = ROOT / "results/SUMMARY.md"; out.write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
