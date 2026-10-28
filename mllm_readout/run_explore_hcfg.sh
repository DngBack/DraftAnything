#!/bin/bash
# exploratory hidden-state guidance (explore_hcfg.py); GPU0 only
cd "$(dirname "$0")"
for m in qwen3vl4b gemma3_4b; do
  for s in illusions mmvp countbench; do
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python explore_hcfg.py --model $m --set $s 2>&1 | grep -v Warn > logs/hcfg_${m}_${s}.log
  done
done
echo ALLDONE >> logs/hcfg_done.log
