#!/bin/bash
# usage: ./run_steer.sh <gpu> <pid-to-wait-for or 0> model ...
cd "$(dirname "$0")"
gpu=$1; w=$2; shift 2
while [ "$w" != 0 ] && kill -0 $w 2>/dev/null; do sleep 60; done
for m in "$@"; do
  for a in 1 16; do
    STEER_ALPHA=$a CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u steer_g3.py --model $m 2>&1 | grep --line-buffered -E "alpha=|Error|Traceback"
  done
done
echo ALLDONE
