#!/bin/bash
# G1.5 (docs/g15_protocol.md). usage: ./run_g15.sh <gpu> "<model family>" ...
cd "$(dirname "$0")"
gpu=$1; shift
for mf in "$@"; do
  set -- $mf
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python pilot.py --model $1 --family $2 --n ${N:-600} --npairs 64 2>&1 | grep -v -E "Loading weights|Warning|tied weights"
done
echo ALLDONE
