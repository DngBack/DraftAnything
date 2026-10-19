#!/bin/bash
# G6 S' and P queue (docs/g6_protocol.md); GPU 0 only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python explore_input.py --src siglip2 dinov2 --sets illusynC 2>&1 | grep -v "Warning\|Loading\|eos_token"
for m in qwen3vl8b qwen35_9b gemma3_4b llava15_7b qwen3vl4b qwen35_4b internvl35_8b qwen2vl_7b; do
  python explore_native.py --model $m --sets illusynC 2>&1 | grep -v "Warning\|Loading"
  G3_DIR=results/g6_syn G3_SETS=illusynC python diag_gen.py --model $m 2>&1 | grep -v "Warning\|Loading"
done
echo SYNCDONE
for m in qwen3vl8b gemma3_4b llava15_7b qwen3vl4b internvl35_8b qwen2vl_7b qwen3vl2b gemma3_27b qwen3vl32b; do
  python explore_native.py --model $m --sets dice tally carlogos 2>&1 | grep -v "Warning\|Loading"
done
echo G6DONE
