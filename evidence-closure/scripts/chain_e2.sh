#!/bin/bash
# start E2 once the 3B E1 run releases its GPU memory
while kill -0 2935092 2>/dev/null; do sleep 30; done
cd /home/bachdx2/tmp/DraftAnything/evidence-closure && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_e2.py --tag qwen25vl7b > results/e2_qwen25vl7b.log 2>&1
