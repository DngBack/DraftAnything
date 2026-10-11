#!/bin/bash
# Sim v2 (division modes): RAFT on GPU1 and genealogy on GPU0, then external eval.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until [ "$(ls outputs/sim/v2/colony_*.pkl 2>/dev/null | wc -l)" -ge 256 ]; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( CUDA_VISIBLE_DEVICES=1 scripts/gpu_retry.sh outputs/logs/raft_sim_v2.log $PY experiments/lineage_raft.py train \
    --train --sim outputs/sim/v2 --no-shape --steps 30000 --workers 6 --tag raft_sim_v2
  CUDA_VISIBLE_DEVICES=1 scripts/gpu_retry.sh outputs/logs/ext_simv2.log $PY experiments/raft_external_eval.py \
    --tags raft_sim_v2 --decoders argmax --out simv2 ) &
( until [ -f outputs/results/sim_genealogy_v2_eval.parquet ]; do sleep 120; done
  CUDA_VISIBLE_DEVICES=0 scripts/gpu_retry.sh outputs/logs/sim_genealogy_v2_v2.log $PY experiments/sim_genealogy.py \
    train --render-v2 --sim v2 --procs 8
  CUDA_VISIBLE_DEVICES=0 scripts/gpu_retry.sh outputs/logs/sim_genealogy_v2_v2_eval.log $PY experiments/sim_genealogy.py \
    eval --render-v2 --sim v2 ) &
wait
