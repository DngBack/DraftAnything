#!/usr/bin/env python3
"""External test set: PHOEBI four-species subset (docs/protocol_external.md).

1. Extract tile features for every phoebi4 image with the same pipeline as phoebi6.
2. Run the pre-registered checks (species correspondence of pure cultures; no duplicates of phoebi6).
3. Build a combined store (phoebi6 rows, then phoebi4 rows) and a manifest whose train/val are the phoebi6
   random split's train/val (all 40 combinations) and whose test is every phoebi4 image (labels mapped to
   the 6 phoebi6 species; bt and mx absent).

    python scripts/phoebi4_prepare.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.features import CACHE, extract, open_store, paths_digest, store_path  # noqa: E402
from popev.phoebi import PHOEBI_ROOT, Manifest  # noqa: E402

MAP = {"b": "bs", "f": "fj", "k": "ka", "p": "pf"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--amendment1", action="store_true",
                    help="mapping confirmed by the dataset card (protocol_external.md, Amendment 1); build despite check 1")
    a = ap.parse_args()
    d4 = json.loads((PHOEBI_ROOT / "data/splits_4class.json").read_text())
    ent4 = [e for part in ("train", "val", "test") for e in d4["splits"][part]]
    p4 = ["data/" + e["path"] for e in ent4]                     # paths relative to the PHOEBI root
    if not a.skip_extract:
        extract(p4, ["cls"], "vit_small_patch14_dinov2.lvd142m", 1, 8, a.device, tag="_p4only")   # variable image sizes: one frame per batch
    F4 = open_store(p4, "cls", "_p4only")

    base = Manifest.default(); K = base.class_names
    F6 = open_store(base.paths, "cls")
    combo6 = np.empty(len(base.paths), dtype=object)
    for v in base.parts.values():
        combo6[v.rows] = v.combos

    # check 1: species correspondence of pure cultures (mean tile embedding, cosine)
    c4 = np.array([e["combo"] for e in ent4])
    mean4 = {c: np.asarray(F4[np.where(c4 == c)[0][:300]], dtype=np.float32).mean((0, 1)) for c in ["b", "f", "k", "p"]}
    pure6 = [k for k in K if (combo6 == k).any()]
    mean6 = {k: np.asarray(F6[np.where(combo6 == k)[0][:300]], dtype=np.float32).mean((0, 1)) for k in pure6}
    cos = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
    report = ["# phoebi4 pre-registered checks", "", "Check 1: nearest phoebi6 pure culture to each phoebi4 pure culture (cosine of mean tile embeddings):", ""]
    ok = True
    for c, m in mean4.items():
        sims = {k: cos(m, mean6[k]) for k in pure6}
        best = max(sims, key=sims.get)
        ok &= best == MAP[c]
        report.append(f"- {c} -> nearest {best} ({sims[best]:.4f}); expected {MAP[c]}; all: " + ", ".join(f"{k} {v:.3f}" for k, v in sims.items()))
    # check 2: duplicates (image-level mean embedding, sample of phoebi4 vs all phoebi6 of the mapped combos)
    rng = np.random.default_rng(0); samp = rng.choice(len(p4), 2000, replace=False)
    A = np.asarray(F4[np.sort(samp)], dtype=np.float32).mean(1); A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = np.asarray(F6[::3], dtype=np.float32).mean(1); B /= np.linalg.norm(B, axis=1, keepdims=True)
    mx = (A @ B.T).max(1)
    report += ["", f"Check 2: max cosine of 2,000 phoebi4 images to 40,000 phoebi6 images: max {mx.max():.5f}, "
               f"99th pct {np.percentile(mx, 99):.5f}; images above 0.999: {(mx > 0.999).sum()}", ""]
    ok &= (mx > 0.999).sum() == 0
    report.append(f"**Checks passed: {ok}**")
    (ROOT / "results/analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/analysis/phoebi4_checks.md").write_text("\n".join(report)); print("\n".join(report))
    if not ok and not a.amendment1:
        sys.exit("pre-registered checks failed; stopping (docs/protocol_external.md)")

    # combined store and manifest
    paths = list(base.paths) + p4
    out = store_path("cls", "_p4")
    S = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(paths),) + F6.shape[1:])
    S[: len(base.paths)] = F6; S[len(base.paths):] = F4; S.flush()
    out.with_suffix(".json").write_text(json.dumps({"kind": "cls", "paths_sha1": paths_digest(paths), "complete": True,
                                                    "n_images": len(paths), "note": "phoebi6 store + phoebi4 rows"}))
    rnd = json.loads((PHOEBI_ROOT / "data/splits.json").read_text())
    lab4 = lambda e: [int(any(MAP[t] == k for t in e["combo"].split("_"))) for k in K]
    man = {"class_names": K, "splits": {
        "train": rnd["splits"]["train"], "val": rnd["splits"]["val"],
        "test": [{"path": "data/" + e["path"], "label": lab4(e), "combo": "p4_" + e["combo"]} for e in ent4]},
        "heldout_combos": sorted({"p4_" + e["combo"] for e in ent4}), "note": "external test: phoebi4"}
    (ROOT / "data/phoebi4").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/phoebi4/splits_lco.json").write_text(json.dumps(man))
    (ROOT / "data/phoebi4/store_paths.json").write_text(json.dumps(paths))
    m = Manifest(ROOT / "data/phoebi4/splits_lco.json", store_paths=paths)
    print({p: len(v) for p, v in m.parts.items()}, "store", S.shape)


if __name__ == "__main__":
    main()
