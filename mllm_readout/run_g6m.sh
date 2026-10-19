#!/bin/bash
# G6 M' queue: same-label swaps at 12.5 / 50% (docs/g6_protocol.md); GPU 0 only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 G6CTRL=1
for m in qwen3vl8b gemma3_4b llava15_7b qwen3vl4b internvl35_8b qwen2vl_7b qwen35_9b qwen35_4b; do
  python -u g5_swap.py --model $m 2>&1 | grep --line-buffered -v "Warning\|Loading"
done
echo G6MDONE
