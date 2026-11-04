#!/usr/bin/env python3
"""Split and data-condition audit (plan §7.1, §8, §17.2). Manifests only; no images, no labels of
held-out combinations are used for anything but counting.

Writes results/audit.md:
  - combinations by order and split, images per combination, leakage checks
  - incidence matrix and species co-occurrence in the trained pool (do any species always co-occur?)
  - per fold: dev combinations, positive/negative dev combinations per species, pure cultures in
    the training pool (prototype eligibility)
  - file-name structure per combination (frame/crop grouping has to be inferred from it)

    python scripts/audit.py [--data_dir DIR] [--out results/audit.md]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.phoebi import PHOEBI_ROOT, Manifest, parse_combo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=PHOEBI_ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit.md")
    a = ap.parse_args()
    man = Manifest(a.data_dir / "splits_lco.json", a.data_dir / "lco_dev_folds.json")
    K = man.class_names
    L = ["# Data audit", "", f"Manifests: `{a.data_dir}`", f"Species: {K}", ""]

    # leakage
    tr, va, te = (set(man.parts[p].combos) for p in ("train", "val", "test"))
    paths = {p: set(man.paths[i] for i in man.parts[p].rows) for p in man.parts}
    L += ["## Leakage checks", "",
          f"- test combos ∩ train combos: {sorted(te & tr) or 'none'}",
          f"- test combos ∩ val combos: {sorted(te & va) or 'none'}",
          f"- train combos == val combos: {tr == va}",
          f"- duplicate image paths across parts: {len(paths['train'] & paths['val']) + len(paths['train'] & paths['test']) + len(paths['val'] & paths['test'])}",
          f"- held-out combos (manifest): {man.heldout}", ""]
    for f, fd in enumerate(man.folds):
        bad = set(fd["dev_combos"] + fd["train_combos"]) & set(man.heldout)
        ov = set(fd["dev_combos"]) & set(fd["train_combos"])
        L.append(f"- fold {f}: touches test combos: {sorted(bad) or 'none'}; dev∩train: {sorted(ov) or 'none'}")
    L.append("")

    # combos
    rows = []
    for c in sorted(tr | te, key=lambda c: (len(parse_combo(c)), c)):
        n = {p: int((man.parts[p].combos == c).sum()) for p in man.parts}
        rows.append(f"| {c} | {len(parse_combo(c))} | {'test' if c in te else 'trained'} | {n['train']} | {n['val']} | {n['test']} |")
    L += ["## Combinations", "", "| combo | order | role | train | val | test |", "|---|---|---|---|---|---|", *rows, ""]

    # incidence / co-occurrence over trained combos
    M = np.array([[k in parse_combo(c) for k in K] for c in man.trained], dtype=int)
    co = M.T @ M
    L += ["## Incidence in the trained pool", "",
          f"{len(man.trained)} trained combinations. Diagonal: positive combinations per species; "
          "off-diagonal: combinations containing both.", "",
          "| | " + " | ".join(K) + " |", "|---|" + "---|" * len(K)]
    L += [f"| {K[i]} | " + " | ".join(str(co[i, j]) for j in range(len(K))) + " |" for i in range(len(K))]
    always = [(K[i], K[j]) for i in range(len(K)) for j in range(len(K)) if i != j and co[i, j] == co[i, i]]
    L += ["", f"Species pairs where the first never appears without the second: {always or 'none'}",
          f"Rank of the incidence matrix: {np.linalg.matrix_rank(M)} (K={len(K)})",
          f"Pure cultures in the trained pool: {[c for c in man.trained if len(parse_combo(c)) == 1]}", ""]

    # folds
    if man.folds:
        L += ["## Development folds", "",
              "| fold | dev combos | " + " | ".join(f"{k} +/-" for k in K) + " | pure cultures in train pool |",
              "|---|---|" + "---|" * len(K) + "---|"]
        for f, fd in enumerate(man.folds):
            dv = fd["dev_combos"]
            pn = [f"{sum(k in parse_combo(c) for c in dv)}/{sum(k not in parse_combo(c) for c in dv)}" for k in K]
            pure = [c for c in fd["train_combos"] if len(parse_combo(c)) == 1]
            L.append(f"| {f} | {len(dv)} | " + " | ".join(pn) + f" | {pure} |")
        alld = [c for fd in man.folds for c in fd["dev_combos"]]
        L += ["", f"Pooled dev: {len(alld)} combinations, {len(set(alld))} distinct; per species positive/negative: "
              + ", ".join(f"{k} {sum(k in parse_combo(c) for c in alld)}/{sum(k not in parse_combo(c) for c in alld)}" for k in K), ""]
        tp = [f"{k} {sum(k in parse_combo(c) for c in man.heldout)}/{sum(k not in parse_combo(c) for c in man.heldout)}" for k in K]
        L += [f"Test: {len(man.heldout)} combinations; positive/negative per species: {', '.join(tp)}", ""]

    # file names
    pat = Counter(re.sub(r"\d+", "N", Path(p).name) for p in man.paths)
    ex = {}
    for p in man.paths:
        ex.setdefault(Path(p).parent.name, []).append(Path(p).name)
    L += ["## File names", "", f"Name patterns: {dict(pat.most_common(5))}", ""]
    for c in list(ex)[:3]:
        nums = sorted(int(m.group()) for n in ex[c] if (m := re.search(r"\d+", n)))
        L.append(f"- {c}: {len(ex[c])} files, index range {nums[0] if nums else '-'}..{nums[-1] if nums else '-'}, e.g. {sorted(ex[c])[:4]}")
    L += ["", "Frame/crop grouping is not encoded in the manifest; the dataset card says each video frame gives "
          "three crops. Verify on images (scripts/audit_images.py) before treating neighbouring indices as one frame.", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
