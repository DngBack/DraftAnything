#!/usr/bin/env python3
"""Development runs: every method of a config on the 5 LCO dev folds, pooled, with intervals.

    python scripts/run_dev.py --config configs/probe.yaml
    python scripts/run_dev.py --config configs/probe.yaml --data_dir data/synth --feature_tag _synth \
        --out results/dev_synth/probe

Never touches the 9 held-out test combinations (Manifest.fold asserts it).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from popev.experiment import run_all, write_report  # noqa: E402
from popev.features import open_store  # noqa: E402
from popev.phoebi import PHOEBI_ROOT, Manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=PHOEBI_ROOT / "data")
    ap.add_argument("--kind", default="cls")
    ap.add_argument("--folds_file", default="lco_dev_folds.json", help="fold file inside --data_dir")
    ap.add_argument("--feature_tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="subset of method names")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    man = Manifest(a.data_dir / "splits_lco.json", a.data_dir / a.folds_file)
    store = open_store(man.paths, a.kind, a.feature_tag)
    out = a.out or ROOT / "results/dev" / f"{a.config.stem}_{a.kind}"
    folds = a.folds if a.folds is not None else list(range(len(man.folds)))
    methods = [m for m in cfg["methods"] if a.methods is None or m["name"] in a.methods]
    run_all(methods, [man.fold(f) for f in folds], store, cfg["n_fields"], cfg["seeds"], out, a.device,
            cfg.get("caps", {}))
    write_report(cfg["methods"], cfg["n_fields"], out, man.class_names, cfg.get("reference", "mean_lin"), a.n_boot,
                 f"Development, pooled dev folds {folds} — {a.config.stem}, units={a.kind}{a.feature_tag}")
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
