#!/bin/bash
# Calibration of image-quality levels on the first 40 attr questions (results kept apart, name=attrcal).
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M3=hf/Qwen3-VL-8B-Instruct
lane() { gpu=$1; tag=$2; model=$3; shift 3; for lv in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -W ignore scripts/run_attr.py --model "$model" --tag $tag --level $lv --limit 40 --name attrcal > results/attrcal_${tag}_$lv.log 2>&1
done; }
M7=hf/Qwen2.5-VL-7B-Instruct
lane 0 qwen25vl7b $M7 s100 s60 s40 &
lane 0 qwen25vl7b $M7 s30 s20 photo &
lane 1 qwen3vl8b $M3 s100 s60 s40 s30 s20 photo &
wait
