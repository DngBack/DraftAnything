#!/bin/bash
# docs/full_lomo_protocol.md -- usage: scripts/full_lomo.sh GPU HELD_OUT
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1; m=$2; tag=raft_v3_lomo_$m
train=$(for x in M0 M1 M2 M3 M4; do [ $x != $m ] && echo -n "$x "; done)
scripts/gpu_retry.sh outputs/logs/$tag.log .venv/bin/python experiments/lineage_raft.py train --tag $tag --seed 0 \
  --train $train --steps 80000 --lr 5e-4 --rot none --init darcy --consist 0.5 --closs 0.3 --family 0.5 --workers 3 || exit 1
scripts/gpu_retry.sh outputs/logs/eval_$tag.log .venv/bin/python experiments/lineage_raft.py eval --tag $tag --eval $m
