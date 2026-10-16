#!/bin/bash
# R1 D (docs/r1_protocol.md): steering controls at alpha=16. usage: ./run_r1_steer.sh <gpu> model...
gpu=$1; shift
mkdir -p logs
for m in "$@"; do for c in random shuffle; do
  CUDA_VISIBLE_DEVICES=$gpu STEER_ALPHA=16 STEER_CTRL=$c .venv/bin/python steer_g3.py --model $m >> logs/r1_steer_$m.log 2>&1
done; done
