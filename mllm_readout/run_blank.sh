#!/bin/bash
cd "$(dirname "$0")"
gpu=$1; shift
for m in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u diag_blank.py --model $m 2>&1 | grep --line-buffered -v -E "Loading weights|Warning|tied weights|Fetching"
done
echo ALLDONE
