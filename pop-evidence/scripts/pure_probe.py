#!/usr/bin/env python3
"""Is the species information in the tiles, and is it masked in mixtures? (plan §17.1)

A 5-way linear classifier on single tiles of the trained pure cultures (bs, fj, ka, mx, pf; bt has
none), fitted on the partition's train images, then:
  (a) in-distribution: tile and image accuracy on the pure cultures' val images. Each pure culture
      is one video, so this is an upper bound that cannot separate species from session.
  (b) on the trained mixtures (orders 2-4; never seen by this classifier, all are LCO dev combos):
      fraction of tiles assigned to each species per combination, and the combination-level AUROC
      of that fraction for the species' presence. High (a) with low (b) = information present in
      isolation but masked or shifted in mixtures.
Test combinations are not loaded.

    python scripts/pure_probe.py [--kind cls]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from popev.features import open_store  # noqa: E402
from popev.metrics import auroc  # noqa: E402
from popev.phoebi import Manifest, parse_combo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="cls")
    ap.add_argument("--tag", default="", help="feature-store tag, e.g. _obj for object-centric units")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    man = Manifest.default(); K = man.class_names
    X = open_store(man.paths, a.kind, a.tag)
    pure = [c for c in man.trained if len(parse_combo(c)) == 1]
    cls = [parse_combo(c)[0] for c in pure]
    dev = torch.device(a.device)

    path_combo = np.empty(len(man.paths), dtype=object)
    for v in man.parts.values():
        path_combo[v.rows] = v.combos

    def load(part, combos):
        v = man.parts[part]; rows = np.sort(v.rows[np.isin(v.combos, combos)])
        return torch.from_numpy(np.asarray(X[rows])).float(), path_combo[rows]

    Xtr, ctr = load("train", pure)
    ytr = torch.tensor([cls.index(parse_combo(c)[0]) for c in ctr])
    T = Xtr.shape[1]
    xt = Xtr.flatten(0, 1).to(dev); yt = ytr.repeat_interleave(T).to(dev)
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    W = torch.nn.Linear(xt.shape[1], len(cls)).to(dev)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(300):
        loss = F.cross_entropy(W((xt - mu) / sd), yt); opt.zero_grad(); loss.backward(); opt.step()
    pred = lambda x: W((x.to(dev) - mu) / sd).softmax(-1).cpu()

    Xva, cva = load("val", pure)
    yva = np.array([cls.index(parse_combo(c)[0]) for c in cva])
    p = pred(Xva.flatten(0, 1)).view(len(Xva), T, -1)
    tile_acc = float((p.argmax(-1).numpy() == yva[:, None]).mean())
    img_acc = float((p.mean(1).argmax(-1).numpy() == yva).mean())
    conf = np.zeros((len(cls), len(cls)), int)
    for t, q in zip(np.repeat(yva, T), p.argmax(-1).flatten().numpy()):
        conf[t, q] += 1
    L = [f"# Pure-culture probe ({a.kind})", "",
         f"Classifier: linear, 5 classes {cls}, fitted on {len(Xtr)} train images ({len(xt)} tiles) of the pure cultures.", "",
         f"(a) Pure-culture val: tile accuracy {tile_acc:.3f}, image accuracy (mean of 16 tile posteriors) {img_acc:.3f}. "
         "Upper bound: one video per species, so species and session coincide.", "",
         "Tile confusion (rows true, columns predicted, row-normalised):", "",
         "| | " + " | ".join(cls) + " |", "|---|" + "---|" * len(cls)]
    for i, k in enumerate(cls):
        L.append(f"| {k} | " + " | ".join(f"{conf[i, j] / conf[i].sum():.2f}" for j in range(len(cls))) + " |")

    mixes = [c for c in man.trained if len(parse_combo(c)) > 1]
    Xm, cm = load("val", mixes)            # val images suffice for fractions; 300 per combination
    fr = pred(Xm.flatten(0, 1)).view(len(Xm), T, -1).argmax(-1)
    frac = np.stack([np.bincount(fr[cm == c].flatten().numpy(), minlength=len(cls)) / fr[cm == c].numel() for c in mixes])
    ymix = np.array([[k in parse_combo(c) for k in cls] for c in mixes])
    L += ["", f"(b) Trained mixtures ({len(mixes)} combinations, orders 2-4, unseen by this classifier): "
          "fraction of tiles assigned to each species.", "",
          "| species | mean fraction when present | when absent | combination AUROC [95 % CI over combinations] |", "|---|---|---|---|"]
    for j, k in enumerate(cls):
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            ii = rng.integers(0, len(mixes), len(mixes)); v = auroc(frac[ii, j], ymix[ii, j])
            if v == v:
                bs.append(v)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L.append(f"| {k} | {frac[ymix[:, j], j].mean():.3f} | {frac[~ymix[:, j], j].mean():.3f} | "
                 f"{auroc(frac[:, j], ymix[:, j]):.2f} [{lo:.2f}, {hi:.2f}]{' resolved' if lo > 0.5 else ''} |")
    dens_path = ROOT / "cache/tile_std.npy"
    if dens_path.exists() and not a.tag:
        D = np.load(dens_path)
        vtr = man.parts["train"]; prow = np.sort(vtr.rows[np.isin(vtr.combos, pure)])
        L += ["", "Content proxy (std of illumination-corrected grey level per tile) of the pure cultures (train):", "",
              "| species | median tile std | share of tiles below global median |", "|---|---|---|"]
        gmed = float(np.median(D))
        for k in cls:
            r = prow[path_combo[prow] == k]
            L.append(f"| {k} | {np.median(D[r]):.2f} | {(D[r] < gmed).mean():.2f} |")
        vm = man.parts["val"]; mrow = np.sort(vm.rows[np.isin(vm.combos, mixes)])
        Dm = D[mrow]
        L += ["", "(c) Tile vote restricted to content tiles (std above a global quantile):", "",
              "| quantile | " + " | ".join(cls) + " |", "|---|" + "---|" * len(cls)]
        argm = fr.numpy()
        for q in (0.0, 0.5, 0.75):
            thr = np.quantile(D, q); keep = Dm > thr
            fq = np.stack([np.bincount(argm[cm == c][keep[cm == c]], minlength=len(cls)) / max(keep[cm == c].sum(), 1) for c in mixes])
            L.append(f"| {q} | " + " | ".join(f"{auroc(fq[:, j], ymix[:, j]):.2f}" for j in range(len(cls))) + " |")
    L += ["", "| combo | " + " | ".join(cls) + " |", "|---|" + "---|" * len(cls)]
    for c, f in sorted(zip(mixes, frac), key=lambda t: (t[0].count("_"), t[0])):
        g = set(parse_combo(c))
        L.append(f"| {c} | " + " | ".join(f"{'✓' if k in g else ' '}{f[j]:.2f}" for j, k in enumerate(cls)) + " |")
    out = ROOT / "results/analysis" / f"pure_probe_{a.kind}{a.tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
