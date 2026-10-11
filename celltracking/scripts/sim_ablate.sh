#!/bin/bash
# Simulator ablations for the sim-to-real claim; runs after the seed runs finish.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f "sim_robust.sh" > /dev/null; do sleep 120; done
until [ "$(ls outputs/sim/v1_fixedphys/colony_*.pkl 2>/dev/null | wc -l)" -ge 256 ]; do sleep 120; done
run() { CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/gpu_retry.sh outputs/logs/$1.log \
          $PY experiments/lineage_raft.py train --train --no-shape --steps 30000 --workers 6 --tag $1 "${@:2}"; }
run raft_sim_fixedphys --sim outputs/sim/v1_fixedphys
run raft_sim_fullfov --sim outputs/sim/v1 --sim-full-fov
run raft_sim_n16 --sim outputs/sim/v1 --n-sim 16
run raft_sim_n64 --sim outputs/sim/v1 --n-sim 64
CUDA_VISIBLE_DEVICES=0 scripts/gpu_retry.sh outputs/logs/ext_simablate.log $PY experiments/raft_external_eval.py \
  --tags raft_sim_fixedphys raft_sim_fullfov raft_sim_n16 raft_sim_n64 --decoders argmax --out simablate
