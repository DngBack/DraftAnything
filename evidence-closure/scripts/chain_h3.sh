#!/bin/bash
cd /home/bachdx2/tmp/DraftAnything/evidence-closure
T="--set h3_train_set.json --split train --qfile tatdqa_dataset_train.json --limit 600"
S="--set e3_test_set.json --split test --qfile tatdqa_dataset_test_gold.json"
( while kill -0 4006334 2>/dev/null; do sleep 60; done
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_h3.py --tag qwen25vl7b $T > results/h3_train_qwen25vl7b.log 2>&1
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -W ignore scripts/run_h3.py --tag qwen25vl7b $S > results/h3_test_qwen25vl7b.log 2>&1 ) &
( while kill -0 4006335 2>/dev/null; do sleep 60; done
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_h3.py --model hf/Qwen3-VL-8B-Instruct --tag qwen3vl8b $T > results/h3_train_qwen3vl8b.log 2>&1
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -W ignore scripts/run_h3.py --model hf/Qwen3-VL-8B-Instruct --tag qwen3vl8b $S --limit 300 > results/h3_test_qwen3vl8b.log 2>&1 ) &
wait
