#!/usr/bin/env python3
"""Lock a test config without git: append its SHA-256 and the lock time; run_test.py refuses to run if the
body changes afterwards. Usage: lock_config.py configs/locked_X.yaml"""
import hashlib, sys, time
from pathlib import Path
p = Path(sys.argv[1]); body = p.read_text()
if "\nlocked_sha256:" in body:
    sys.exit("already locked")
body = body.rstrip("\n") + "\n"
p.write_text(body + f"\nlocked_sha256: {hashlib.sha256(body.encode()).hexdigest()}\nlocked_at: '{time.strftime('%Y-%m-%d %H:%M:%S')}'\n")
log = Path(__file__).resolve().parents[1] / "reports/experiment_log.md"
with log.open("a") as f:
    f.write(f"\n- {time.strftime('%Y-%m-%d %H:%M')} LOCKED `{p.name}` sha256 {hashlib.sha256(body.encode()).hexdigest()[:12]}\n")
print("locked", p)
