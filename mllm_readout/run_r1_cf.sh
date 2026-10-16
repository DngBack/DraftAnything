#!/bin/bash
# R1 B + C1 (docs/r1_protocol.md). usage: ./run_r1_cf.sh <gpu> model...
gpu=$1; shift
mkdir -p logs
for m in "$@"; do CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python cf_pairs.py --model $m >> logs/r1_cf_$m.log 2>&1; done
