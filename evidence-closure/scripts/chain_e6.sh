#!/bin/bash
cd /home/bachdx2/tmp/DraftAnything/evidence-closure
while kill -0 2291914 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_e6.py --tag qwen25vl7b > results/e6_qwen25vl7b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_e6.py --model hf/Qwen2.5-VL-3B-Instruct --tag qwen25vl3b > results/e6_qwen25vl3b.log 2>&1 &
wait
