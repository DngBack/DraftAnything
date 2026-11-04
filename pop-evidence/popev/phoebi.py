"""PHOEBI manifests, the LCO partition and its development folds, as row indices into the feature store.

Facts this module relies on (read from the PHOEBI code at commit 2a42be0, not yet from the data):
- one video per combination: `combo` is at once the label set, the culture and the acquisition
  session, so "multi-field" bags can only be drawn from the frames of one video;
- `splits_lco.json`: train/val are an image-level 90/10 split of the 31 trained-on combinations,
  test is every image of the 9 held-out combinations;
- `lco_dev_folds.json`: 5 folds over the 26 trained-on combinations of order 2-4; the 5 trained
  singletons stay in every fold's training pool (bt is the held-out singleton, so bt never has a
  pure culture during development or test).

Rules (protocol.md §3): train on the fold's train_combos ∩ partition train; calibrate thresholds and
hyper-parameters that need in-distribution data on train_combos ∩ partition val; evaluate on every
image (train+val) of the dev_combos. The test view is only built by scripts/run_test.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PHOEBI_ROOT = Path(os.environ.get("PHOEBI_ROOT", "/home/jovyan/bachdx2/datasets/phoebi/repo"))


def parse_combo(combo: str) -> List[str]:
    return [t for t in combo.split("_") if t]


@dataclass
class View:
    rows: np.ndarray          # (N,) indices into the feature store
    labels: np.ndarray        # (N, K) int64
    combos: np.ndarray        # (N,) str

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, mask: np.ndarray) -> "View":
        return View(self.rows[mask], self.labels[mask], self.combos[mask])

    @staticmethod
    def concat(views: Sequence["View"]) -> "View":
        return View(np.concatenate([v.rows for v in views]), np.concatenate([v.labels for v in views]),
                    np.concatenate([v.combos for v in views]))


@dataclass
class Split:
    train: View      # fit the decoder
    calib: View      # thresholds, early stopping; in-distribution
    eval: View       # held-out combinations (dev or test)
    name: str


class Manifest:
    """The LCO partition with every image mapped to a row of the feature store."""

    def __init__(self, splits_path: Path, folds_path: Path | None = None, store_paths: List[str] | None = None) -> None:
        """store_paths: the path order of an existing feature store (another partition's manifest);
        rows then index that store instead of this manifest's own order."""
        d = json.loads(Path(splits_path).read_text())
        self.class_names: List[str] = list(d["class_names"])
        self.paths: List[str] = []
        parts: Dict[str, View] = {}
        for part in ("train", "val", "test"):
            entries = d["splits"][part]
            start = len(self.paths)
            self.paths.extend(e["path"] for e in entries)
            parts[part] = View(np.arange(start, len(self.paths)),
                               np.asarray([e["label"] for e in entries], dtype=np.int64).reshape(len(entries), len(self.class_names)),
                               np.asarray([e.get("combo", e.get("video")) for e in entries]))
        if store_paths is not None:
            pos = {p: i for i, p in enumerate(store_paths)}
            remap = np.array([pos[p] for p in self.paths])
            for v in parts.values():
                v.rows = remap[v.rows]
            self.paths = list(store_paths)
        self.parts = parts
        self.heldout = sorted(d.get("heldout_combos") or set(parts["test"].combos))
        self.trained = sorted(d.get("trained_combos") or set(parts["train"].combos))
        self.folds = json.loads(Path(folds_path).read_text())["folds"] if folds_path else []

    @classmethod
    def default(cls) -> "Manifest":
        return cls(PHOEBI_ROOT / "data/splits_lco.json", PHOEBI_ROOT / "data/lco_dev_folds.json")

    @classmethod
    def partition(cls, seed: int, data_dir: Path = PHOEBI_ROOT / "data") -> "Manifest":
        """LCO partition `seed`, rows into the feature store built from the seed-1337 manifest."""
        base = cls(data_dir / "splits_lco.json")
        if seed == 1337:
            return base
        return cls(data_dir / f"splits_lco_seed{seed}.json", store_paths=base.paths)

    def _in(self, part: str, combos: Sequence[str]) -> View:
        v = self.parts[part]
        return v.select(np.isin(v.combos, list(combos)))

    def fold(self, f: int) -> Split:
        fd = self.folds[f]
        dev, train = fd["dev_combos"], fd["train_combos"]
        assert not set(dev) & set(train), "dev and train combos overlap"
        assert not set(dev + train) & set(self.heldout), "a fold touches a held-out test combination"
        if "calib_combos" in fd:
            # datasets whose grouping unit is the image (BBBC041): calibration is its own set of units
            cal = fd["calib_combos"]
            assert not set(cal) & (set(dev) | set(train)), "calibration overlaps train or dev"
            both = lambda cs: View.concat([self._in("train", cs), self._in("val", cs)])
            return Split(both(train), both(cal), both(dev), f"fold{f}")
        return Split(self._in("train", train), self._in("val", train),
                     View.concat([self._in("train", dev), self._in("val", dev)]), f"fold{f}")

    def test_split(self, name: str = "test") -> Split:
        if len(self.parts["val"]):
            return Split(self.parts["train"], self.parts["val"], self.parts["test"], name)
        # datasets without a val part (unit = image): calibrate on a fixed 20 % of the training units,
        # chosen by a hash of the unit id so the choice does not depend on order or on the method
        import hashlib
        tr = self.parts["train"]
        cal = np.array([int(hashlib.md5(c.encode()).hexdigest(), 16) % 5 == 0 for c in tr.combos])
        return Split(tr.select(~cal), tr.select(cal), self.parts["test"], name)
