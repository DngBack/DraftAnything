#!/bin/bash
# G3 (docs/g3_protocol.md). usage: ./run_g3.sh <gpu> <log tag> model ...
cd "$(dirname "$0")"
gpu=$1; shift
for m in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python diag_gen.py --model $m 2>&1 | grep -v -E "Loading weights|Warning|tied weights|Fetching"
done
echo ALLDONE
