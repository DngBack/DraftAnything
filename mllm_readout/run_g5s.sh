#!/bin/bash
# G5 S queue (docs/g5_protocol.md): illusyn G3 states, external and native features; GPU 0 only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python explore_input.py --src siglip2 dinov2 --sets illusyn 2>&1 | grep -v "Warning\|Loading"
for m in qwen3vl8b qwen35_9b gemma3_4b llava15_7b qwen3vl4b qwen35_4b internvl35_8b qwen2vl_7b; do
  G3_DIR=results/g5_syn G3_SETS=illusyn python diag_gen.py --model $m 2>&1 | grep -v "Warning\|Loading"
  python explore_native.py --model $m --sets illusyn 2>&1 | grep -v "Warning\|Loading"
done
echo SYNDONE
