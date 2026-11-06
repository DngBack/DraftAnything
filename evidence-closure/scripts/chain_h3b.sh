#!/bin/bash
# H3b test2 data collection (same actions as H3). Sequential on GPU 1.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="--set h3b_test2_set.json --split test --name test2 --qfile tatdqa_dataset_test_gold.json"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_h3.py --tag qwen25vl7b $S > results/h3_test2_qwen25vl7b.log 2>&1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_h3.py --model hf/Qwen3-VL-8B-Instruct --tag qwen3vl8b $S > results/h3_test2_qwen3vl8b.log 2>&1
