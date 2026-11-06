#!/bin/bash
cd "$(dirname "$0")/.."
A="--set e3_test_set.json --split test --prefix final"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_final.py --tag qwen25vl7b $A --bs 10 > results/final_qwen25vl7b.log 2>&1 &
P7=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_final.py --model hf/Qwen2.5-VL-3B-Instruct --tag qwen25vl3b $A --bs 10 > results/final_qwen25vl3b.log 2>&1 &
P3=$!
wait -n $P7 $P3
GPU=0; kill -0 $P7 2>/dev/null && GPU=1
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -W ignore scripts/run_final.py --model hf/Qwen3-VL-8B-Instruct --tag qwen3vl8b $A --bs 8 --limit 300 > results/final_qwen3vl8b.log 2>&1
wait
