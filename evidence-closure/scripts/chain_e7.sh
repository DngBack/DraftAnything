#!/bin/bash
cd /home/bachdx2/tmp/DraftAnything/evidence-closure
while kill -0 3095859 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_e6.py --model hf/Qwen3-VL-8B-Instruct --tag qwen3vl8b --set e3_test_set.json --split test --prefix e6test --bs 10 > results/e6test_qwen3vl8b.log 2>&1
