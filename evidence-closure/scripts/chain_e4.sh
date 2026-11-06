#!/bin/bash
# run E4 on each GPU as soon as that GPU's E3 run ends
cd /home/bachdx2/tmp/DraftAnything/evidence-closure
( while kill -0 2169294 2>/dev/null; do sleep 30; done
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_e4.py --tag qwen25vl7b > results/e4_qwen25vl7b.log 2>&1 ) &
( while kill -0 2169295 2>/dev/null; do sleep 30; done
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_e4.py --model hf/Qwen2.5-VL-3B-Instruct --tag qwen25vl3b > results/e4_qwen25vl3b.log 2>&1 ) &
wait
