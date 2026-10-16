#!/bin/bash
# Addendum 4 (docs/g3_protocol.md): held-out models -> extraction (illusions, CountBench) + steering at alpha 1 and 16.
# usage: ./run_heldout.sh <gpu> model...
gpu=$1; shift
export CUDA_VISIBLE_DEVICES=$gpu G3_DIR=results/g3_heldout G3_SETS=illusions,countbench
mkdir -p $G3_DIR logs
for m in "$@"; do
  .venv/bin/python diag_gen.py --model $m >> logs/heldout_$m.log 2>&1
  for a in 1 16; do STEER_ALPHA=$a .venv/bin/python steer_g3.py --model $m >> logs/heldout_$m.log 2>&1; done
done
