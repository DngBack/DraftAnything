#!/bin/bash
# G7 Addendum A: grid features of the own LLM input (layer 0) on illusynC / illusynP; GPU 0 only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
for m in qwen3vl8b qwen3vl4b gemma3_4b llava15_7b internvl35_8b qwen2vl_7b qwen35_9b qwen35_4b; do
  python explore_input.py --src $m --sets illusynC illusynP 2>&1 | grep -v "Warning\|Loading"
done
echo G7GDONE
