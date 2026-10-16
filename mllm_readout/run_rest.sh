#!/bin/bash
# finish G1: gemma spatial (interrupted) + qwen reruns with calibration control
cd "$(dirname "$0")"
for mf in "gemma3_4b spatial" "qwen3vl4b count" "qwen3vl4b spatial"; do
  set -- $mf
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python pilot.py --model $1 --family $2 --n 600 --npairs 64 2>&1 | grep -v -E "Loading weights|Warning"
done
echo ALLDONE
