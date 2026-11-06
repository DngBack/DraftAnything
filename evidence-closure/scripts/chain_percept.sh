#!/bin/bash
# Pilot B on both models, one process per GPU.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_percept.py --model hf/Qwen2.5-VL-7B-Instruct --tag qwen25vl7b > results/percept_qwen25vl7b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_percept.py --model hf/Qwen3-VL-8B-Instruct --tag qwen3vl8b > results/percept_qwen3vl8b.log 2>&1 &
wait
