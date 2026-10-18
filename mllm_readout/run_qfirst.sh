#!/bin/bash
# R6 question-first (exploratory, 2026-10-18): full sets with free-generation control, then the mechanism probe
# (question-aware image tokens at the probe depth). GPU0, small batches (shared GPU).
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
$P explore_input.py --deep --qfirst --src qwen3vl4b qwen35_4b --sets countbench illusions
for m in qwen3vl4b qwen35_4b qwen3vl8b qwen35_2b qwen35_9b; do for mode in qfirst plain; do
  $P explore_repair.py --model $m --mode $mode --sets countbench illusions mmvp --bs 16; done; done
$P explore_input.py --deep --qfirst --src qwen3vl8b qwen35_2b qwen35_9b --sets countbench illusions
echo QFIRSTDONE
