#!/usr/bin/env bash
# Detached (setsid) launch of the pending development runs, so they survive the shell/session that
# started them. Safe to re-run: run_dev.py skips prediction files that already exist. Logs in logs/.
#   GPU=0 bash scripts/launch_all.sh
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
L() { setsid nohup bash -c "$1" > "logs/$2.log" 2>&1 < /dev/null & }
PY=".venv/bin/python"
L "CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_dev.py --config configs/probe.yaml --kind cls && CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_dev.py --config configs/probe.yaml --kind patchmean" probe
L "CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_dev.py --config configs/popevidence.yaml --kind cls --n_boot 1000" popevidence_cls
L "CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_dev.py --config configs/popevidence_tile.yaml --kind cls --n_boot 1000" popevidence_tile_cls
L "CUDA_VISIBLE_DEVICES=$GPU bash scripts/repro_phoebi.sh devcv" repro_devcv
