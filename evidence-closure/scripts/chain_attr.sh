#!/bin/bash
# Full attribution data: 360 questions (first 40 were calibration), all locked levels, both models.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M7=hf/Qwen2.5-VL-7B-Instruct; M3=hf/Qwen3-VL-8B-Instruct
lane() { gpu=$1; tag=$2; model=$3; shift 3; for x in "$@"; do lv=${x%:*}; op=""; [ "${x#*:}" = op ] && op=--op
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -W ignore scripts/run_attr.py --model "$model" --tag $tag --level $lv --skip 40 $op > results/attr_${tag}_$lv.log 2>&1
done; }
lane 0 qwen25vl7b $M7 s100:op s50 s30:op &
lane 0 qwen25vl7b $M7 s60 s40 photo &
lane 1 qwen3vl8b $M3 s100:op s50 s30:op &
lane 1 qwen3vl8b $M3 s60 s40 photo &
wait
