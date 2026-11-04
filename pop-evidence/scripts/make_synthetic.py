#!/usr/bin/env python3
"""A synthetic stand-in for PHOEBI with a planted "beyond-the-mean" signal, to test the pipeline.

Same structure as the real manifests: 6 species, 40 combinations (6/15/10/8/1 by order 1/2/3/4/6),
the LCO hold-out counts {1:1, 2:2, 3:3, 4:2, 6:1} with `bt` as the held-out singleton, an image-level
90/10 train/val split, and 5 dev folds over the trained combinations of order 2-4.

Tiles: each tile draws one species of its combination and samples z = mu_k + noise + session_c,
then L2-normalises. Species bs and bt share mu but bs is tight and bt bimodal (mu +/- delta*u), so
the mean of a bag cannot tell them apart and its spread can. Every combination has its own session
offset, as one video per combination in PHOEBI. Expected: mean pooling near chance for bs/bt,
spread-aware pooling above it; other species separable by the mean.

    python scripts/make_synthetic.py --out data/synth   # writes manifests + cache/features_cls_synth.npy
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from popev.features import paths_digest, store_path  # noqa: E402

K = ["bs", "bt", "fj", "ka", "mx", "pf"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/synth")
    ap.add_argument("--images_per_combo", type=int, default=300)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--delta", type=float, default=1.2)
    ap.add_argument("--session", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    combos = [(k,) for k in K]
    for r, n in ((2, 15), (3, 10), (4, 8)):
        allc = list(itertools.combinations(K, r))
        combos += [allc[i] for i in sorted(rng.choice(len(allc), n, replace=False))]
    combos.append(tuple(K))
    names = ["_".join(c) for c in combos]
    by_order = {}
    for c in combos:
        by_order.setdefault(len(c), []).append("_".join(c))
    while True:
        held = ["bt"] + [str(c) for o, n in ((2, 2), (3, 3), (4, 2)) for c in rng.choice(by_order[o], n, replace=False)] + [names[-1]]
        trained = [c for c in names if c not in held]
        if all(any(k in c.split("_") for c in trained) for k in K):
            break

    D = a.dim
    mu = rng.normal(size=(len(K), D)); mu[1] = mu[0]                 # bs, bt share the mean
    u = rng.normal(size=D); u /= np.linalg.norm(u)
    lab = lambda c: [int(k in c.split("_")) for k in K]
    splits = {"train": [], "val": [], "test": []}
    feats = []
    for c in names:
        sess = rng.normal(size=D) * a.session
        sp = [K.index(k) for k in c.split("_")]
        for i in range(a.images_per_combo):
            ks = rng.choice(sp, 16)
            z = mu[ks] + rng.normal(size=(16, D)) * 0.6 + sess
            bt = ks == 1
            z[bt] += np.where(rng.random(bt.sum()) < 0.5, -1, 1)[:, None] * a.delta * u * np.sqrt(D) / 4
            feats.append(z / np.linalg.norm(z, axis=1, keepdims=True))
            e = {"path": f"data/images/{c}/{i}.jpg", "label": lab(c), "combo": c}
            if c in held:
                splits["test"].append(e)
            else:
                splits["train" if rng.random() > 0.1 else "val"].append(e)

    a.out.mkdir(parents=True, exist_ok=True)
    d = {"class_names": K, "heldout_combos": sorted(held), "trained_combos": sorted(trained), "splits": splits,
         "seed": a.seed, "synthetic": True}
    (a.out / "splits_lco.json").write_text(json.dumps(d))
    pool = [c for c in trained if len(c.split("_")) in (2, 3, 4)]
    pool = [pool[i] for i in rng.permutation(len(pool))]
    folds = [{"fold": f, "dev_combos": sorted(pool[f::5]),
              "train_combos": sorted(c for c in trained if c not in pool[f::5])} for f in range(5)]
    (a.out / "lco_dev_folds.json").write_text(json.dumps({"folds": folds, "heldout_test_combos": sorted(held)}))

    # feature rows must follow Manifest order: train, val, test
    path_to_row = {}
    idx = 0
    for c in names:
        for i in range(a.images_per_combo):
            path_to_row[f"data/images/{c}/{i}.jpg"] = idx; idx += 1
    paths = [e["path"] for p in ("train", "val", "test") for e in splits[p]]
    F = np.stack(feats)[[path_to_row[p] for p in paths]].astype(np.float16)
    p = store_path("cls", "_synth"); p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, F)
    p.with_suffix(".json").write_text(json.dumps({"kind": "cls", "synthetic": True, "paths_sha1": paths_digest(paths),
                                                  "complete": True, "n_images": len(paths)}))
    print(f"{len(names)} combos, held out {sorted(held)}; features {F.shape} -> {p}")


if __name__ == "__main__":
    main()
