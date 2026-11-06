#!/bin/bash
cd /home/bachdx2/tmp/DraftAnything/evidence-closure
( while kill -0 2244786 2>/dev/null; do sleep 30; done
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_e5.py --tag qwen25vl7b > results/e5_qwen25vl7b.log 2>&1
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_e5.py --tag qwen25vl7b --set e3_test_set.json --split test --prefix e5test > results/e5test_qwen25vl7b.log 2>&1 ) &
( while kill -0 2244787 2>/dev/null; do sleep 30; done
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_e5.py --model hf/Qwen2.5-VL-3B-Instruct --tag qwen25vl3b > results/e5_qwen25vl3b.log 2>&1
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_e5.py --model hf/Qwen2.5-VL-3B-Instruct --tag qwen25vl3b --set e3_test_set.json --split test --prefix e5test > results/e5test_qwen25vl3b.log 2>&1 ) &
wait
