#!/bin/bash
# G5 GPU queue (docs/g5_protocol.md), GPU 0 only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
for m in qwen3vl8b qwen35_9b gemma3_4b llava15_7b internvl35_8b qwen2vl_7b qwen3vl4b qwen35_4b qwen3vl2b qwen2vl_2b qwen35_2b gemma3_27b qwen3vl32b; do
  python explore_native.py --model $m --sets illusions countbench 2>&1 | grep -v "Warning\|Loading"
done
echo NATIVEDONE
for m in qwen3vl8b qwen35_9b gemma3_4b llava15_7b qwen3vl4b qwen35_4b internvl35_8b qwen2vl_7b; do
  python g5_swap.py --model $m 2>&1 | grep -v "Warning\|Loading"
done
echo SWAPDONE
