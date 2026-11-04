"""Run a list of methods over splits x n_fields x seeds, save predictions, write the report.

Development: splits = the 5 dev folds; predictions of each fold's dev combinations are pooled into
one set of 26 combinations (each held out once), the development analogue of the test split.
Test: splits = [the LCO test split]; only scripts/run_test.py builds it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from popev import metrics
from popev.phoebi import Split
from popev.train import run


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_all(methods: List[Dict], splits: List[Split], store, n_fields: List[int], seeds: List[int],
            out: Path, device: str, caps: Dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for m in methods:
        for n in n_fields:
            f = out / f"preds_{m['name']}_n{n}.npz"
            if f.exists():
                log(f"skip {f.name} (exists)"); continue
            S, P, n_params, thr, A = [], [], 0, [], []
            for seed in seeds:
                parts = []
                for sp in splits:
                    t0 = time.time()
                    r = run(m, sp, store, n, seed, device, **caps)
                    log(f"{m['name']} n={n} seed={seed} {sp.name}: {len(r['Y'])} bags, "
                        f"AUROC {np.nanmean([metrics.auroc(r['S'][:, k], r['Y'][:, k]) for k in range(r['Y'].shape[1])]):.3f} "
                        f"({time.time() - t0:.0f}s)")
                    parts.append(r); n_params = r["n_params"]
                S.append(np.concatenate([r["S"] for r in parts])); P.append(np.concatenate([r["P"] for r in parts]))
                thr.append(np.stack([r["thr"] for r in parts]))
                if parts[0]["A"] is not None:
                    A.append(np.concatenate([r["A"] for r in parts]))
            Y = np.concatenate([r["Y"] for r in parts])
            # several test partitions share combinations; their unit is "partition:combination"
            multi = len(splits) > 1 and not splits[0].name.startswith("fold")
            C = np.concatenate([np.char.add(f"{sp.name}:", r["C"].astype(str)) if multi else r["C"]
                                for sp, r in zip(splits, parts)])
            # store row of every bag's images in the feature store, for tile-level analyses
            R = np.concatenate([r["rows"][r["pos"]] for r in parts])
            extra = {"A": np.stack(A)} if A else {}
            G = np.concatenate([np.full(len(r["Y"]), i) for i, r in enumerate(parts)])   # split (fold / partition) of each bag
            np.savez_compressed(f, S=np.stack(S), P=np.stack(P), Y=Y, C=C, thr=np.stack(thr), R=R, G=G,
                                n_params=n_params, seeds=np.array(seeds), **extra)


def load_preds(out: Path, name: str, n: int):
    d = np.load(out / f"preds_{name}_n{n}.npz")
    return d["S"], d["P"], d["Y"], d["C"], int(d["n_params"])


def write_report(methods: List[Dict], n_fields: List[int], out: Path, class_names: List[str],
                 reference: str, n_boot: int, title: str) -> Dict:
    f = metrics.fmt
    lines = [f"# {title}", "",
             "Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. "
             "Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. "
             "Thresholds: per-species argmax-F1 on in-distribution calibration bags.", ""]
    summary: Dict = {}
    for n in n_fields:
        lines += [f"## n_fields = {n}", "",
                  "| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |",
                  "|---|---|---|---|---|---|---|---|"]
        rows, ref = {}, None
        for m in methods:
            p = out / f"preds_{m['name']}_n{n}.npz"
            if not p.exists():
                continue
            S, P, Y, C, npar = load_preds(out, m["name"], n)
            r = metrics.report(S, Y, P, C, class_names, n_boot=n_boot)
            z = np.load(out / f"preds_{m['name']}_n{n}.npz")
            if "G" in z.files and len(np.unique(z["G"])) > 1:
                Sr = metrics.rank_within(S, z["G"])
                r["auroc_split_ranks"] = metrics.report(Sr, Y, P, C, class_names, n_boot=n_boot)["auroc_mean"]
            r["context"] = metrics.context_sensitivity(S.mean(0), Y, C, class_names)
            r["attribution"] = metrics.context_attribution(S.mean(0), Y, C, n_boot=n_boot)
            r["by_combination"] = metrics.by_combination(S.mean(0), Y, P[0], C, class_names)
            rows[m["name"]] = (S, P, Y, C, r)
            lines.append(f"| {m['name']} | {f(r['macro_ap'])} | {f(r['auroc_mean'])} | {f(r['f1_margin'])} | "
                         f"{f(r['macro_f1'])} | {r['fpr_mean']['point']:.3f} | {r['n_pred']['point']:.2f} / "
                         f"{Y.sum(1).mean():.2f} | {npar} |"
                         + (f" split-rank AUROC {f(r['auroc_split_ranks'])}" if "auroc_split_ranks" in r else ""))
        if not rows:
            continue
        lines += ["", "Per-species AP (AUROC):", "",
                  "| method | " + " | ".join(class_names) + " |", "|---|" + "---|" * len(class_names)]
        for name, (*_, r) in rows.items():
            ps = r["per_species"]
            lines.append(f"| {name} | " + " | ".join(f"{ps[k]['ap']:.2f} ({ps[k]['auroc']:.2f})" for k in class_names) + " |")
        any_r = next(iter(rows.values()))[-1]["per_species"]
        lines.append("| pos/neg combos | " + " | ".join(f"{any_r[k]['n_pos_combos']}/{any_r[k]['n_neg_combos']}" for k in class_names) + " |")
        if reference in rows:
            S0, P0, Y0, C0, _ = rows[reference]
            lines += ["", f"Paired difference against `{reference}` (same bags, same combination resamples):", "",
                      "| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |", "|---|---|---|---|---|"]
            for name, (S, P, Y, C, r) in rows.items():
                if name == reference:
                    continue
                d = metrics.paired_delta((S, P), (S0, P0), Y, C, n_boot=n_boot)
                r["delta_vs_ref"] = d
                lines.append(f"| {name} | {f(d['macro_ap'])} | {d['macro_ap']['p_gt0']:.2f} | {f(d['auroc_mean'])} | {f(d['fpr_mean'])} |")
        lines += ["", "Context sensitivity (per-combination mean score: spread over positive combos S_pos, "
                  "positive-negative gap, combination-level AUROC):", "",
                  "| method | " + " | ".join(class_names) + " |", "|---|" + "---|" * len(class_names)]
        for name, (*_, r) in rows.items():
            c = r["context"]
            lines.append(f"| {name} | " + " | ".join(f"{c[k]['S_pos']:.2f} / {c[k]['gap']:+.2f} / {c[k]['combo_auroc']:.2f}" for k in class_names) + " |")
        lines += ["", "Own/context attribution (logit of per-combination mean score regressed on the presence vector; "
                  "own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 "
                  "(resolved: CI below 0), positive resolved = CI above 0):", "",
                  "| method | " + " | ".join(f"own {k}" for k in class_names) + " | context | inverted (resolved) | positive resolved |",
                  "|---|" + "---|" * (len(class_names) + 3)]
        for name, (*_, r) in rows.items():
            a = r["attribution"]
            lines.append(f"| {name} | " + " | ".join(f"{v:+.2f} [{c[0]:+.1f}, {c[1]:+.1f}]" for v, c in zip(a["own"], a["own_ci95"]))
                         + f" | {a['context_mean_abs']:.2f} | {a['n_inverted']} ({a['inverted_resolved']}) | {a['positive_resolved']} |")
        lines.append("")
        summary[n] = {name: r for name, (*_, r) in rows.items()}
    (out / "report.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
