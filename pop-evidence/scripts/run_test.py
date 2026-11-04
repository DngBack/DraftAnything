#!/usr/bin/env python3
"""The one-time evaluation on the 9 held-out LCO combinations.

Runs only a config marked `locked: true` whose `locked_at_commit` is an ancestor of HEAD and whose
methods/hyper-parameters are unchanged since that commit (the file must be committed and clean).
Refuses to overwrite an existing results/test/<name>/ directory: a second look at the test set is
a new, logged decision (--allow_rerun "<reason>"), recorded in reports/experiment_log.md.

    python scripts/run_test.py --config configs/locked.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from popev.experiment import run_all, write_report  # noqa: E402
from popev.features import open_store  # noqa: E402
from popev.phoebi import PHOEBI_ROOT, Manifest  # noqa: E402


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=PHOEBI_ROOT / "data")
    ap.add_argument("--kind", default="cls")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--allow_rerun", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if not (cfg.get("locked") or cfg.get("locked_sha256")):
        sys.exit("config is not locked; select on dev folds, then lock it (protocol.md §6)")
    if "locked_sha256" in cfg:
        # hash lock (no git): scripts/lock_config.py wrote the digest of the config body at lock time
        body = a.config.read_text().split("\nlocked_sha256:")[0]
        if hashlib.sha256(body.encode()).hexdigest() != cfg["locked_sha256"]:
            sys.exit("config changed after it was locked")
    else:
        if git("status", "--porcelain", str(a.config)).stdout.strip():
            sys.exit("locked config has uncommitted changes")
        if git("merge-base", "--is-ancestor", cfg["locked_at_commit"], "HEAD").returncode != 0:
            sys.exit("locked_at_commit is not an ancestor of HEAD")
        if git("diff", "--quiet", cfg["locked_at_commit"], "HEAD", "--", str(a.config)).returncode != 0:
            sys.exit("config changed after the lock commit")
    out = ROOT / "results/test" / cfg["name"]
    if out.exists() and not a.allow_rerun:
        sys.exit(f"{out} exists; the test set has been read for this config")
    ds = cfg.get("dataset", "phoebi")
    if ds == "phoebi":
        parts = cfg.get("partitions", [1337])
        mans = {p: Manifest.partition(p, a.data_dir) for p in parts}
        man = mans[parts[0]]
        store = open_store(man.paths, a.kind)
        splits = [mans[p].test_split(f"lco{p}") for p in parts]
    elif ds == "phoebi4":
        # external test (docs/protocol_external.md): train/calibrate on phoebi6, test on the phoebi4 videos
        import json
        store_paths = json.loads((ROOT / "data/phoebi4/store_paths.json").read_text())
        parts = [ds]
        man = Manifest(ROOT / "data/phoebi4/splits_lco.json", store_paths=store_paths)
        store = open_store(man.paths, a.kind, cfg.get("store_tag", "_p4"))
        splits = [man.test_split("phoebi4")]
    else:
        parts = [ds]
        man = Manifest(ROOT / f"data/{ds}/splits_lco.json")
        store = open_store(man.paths, a.kind, f"_{ds}")
        splits = [man.test_split(f"{ds}_test")]
    log = ROOT / "reports/experiment_log.md"
    with log.open("a") as f:
        f.write(f"\n- {time.strftime('%Y-%m-%d %H:%M')} TEST RUN `{cfg['name']}` config@{cfg.get('locked_at_commit') or 'sha256:' + str(cfg.get('locked_sha256'))[:12]} "
                f"HEAD@{git('rev-parse', '--short', 'HEAD').stdout.strip()}"
                + (f" RERUN: {a.allow_rerun}" if a.allow_rerun else "") + "\n")
    run_all(cfg["methods"], splits, store, cfg["n_fields"], cfg["seeds"], out, a.device, cfg.get("caps", {}))
    write_report(cfg["methods"], cfg["n_fields"], out, man.class_names, cfg.get("reference", "mean_lin"), a.n_boot,
                 f"Held-out test {parts} (read once) — {cfg['name']}")
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
