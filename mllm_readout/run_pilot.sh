#!/bin/bash
# Pilot (exploratory, 2026-10-18): G3 errors + 40 correct per cell; one free-generation control and each repair on the same items.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
P=.venv/bin/python
for m in qwen3vl4b qwen35_4b; do for mode in plain qfirst res2; do
  $P explore_repair.py --model $m --mode $mode --sets illusions countbench --pilot 40; done; done
echo PILOTDONE
