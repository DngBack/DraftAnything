#!/bin/bash
# Fully zero-shot OIAM pipeline: sim-trained LineageRAFT + sim-trained genealogy decoder.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until [ -f outputs/results/sim_genealogy_eval.parquet ]; do sleep 120; done
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
scripts/gpu_retry.sh outputs/logs/imgpart_sim_simgen.log $PY experiments/image_partition_decode.py --tag raft_sim_loc --genealogy sim
scripts/gpu_retry.sh outputs/logs/imgpart_sim_realgen.log $PY experiments/image_partition_decode.py --tag raft_sim_loc --genealogy real
scripts/gpu_retry.sh outputs/logs/imgpart_v3_simgen.log $PY experiments/image_partition_decode.py --tag raft_v3 --genealogy sim
