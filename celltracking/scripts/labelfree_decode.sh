#!/bin/bash
# Fully label-free pipeline: sim-trained RAFT + sim-v2-trained genealogy in the Potts decoder.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -f outputs/results/glue/raft_sim_v2.pt ]; do sleep 60; done
for tag in raft_sim_loc raft_sim_v2; do
  scripts/gpu_retry.sh outputs/logs/imgpart_${tag}_simgenv2.log $PY experiments/image_partition_decode.py \
    --tag $tag --genealogy sim --gen-ckpt sim_genealogy_v2_v2
done
scripts/gpu_retry.sh outputs/logs/imgpart_v3_simgenv2.log $PY experiments/image_partition_decode.py \
  --tag raft_v3 --genealogy sim --gen-ckpt sim_genealogy_v2_v2
