#!/bin/bash
# G7 queue (docs/g7_protocol.md): illusynP features, G3 decision step, text-only comparison; GPU 0 only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python explore_input.py --src siglip2 dinov2 --sets illusynP 2>&1 | grep -v "Warning\|Loading\|eos_token"
for m in qwen3vl8b qwen35_9b gemma3_4b llava15_7b qwen3vl4b qwen35_4b internvl35_8b qwen2vl_7b; do
  python g7_text.py --model $m 2>&1 | grep -v "Warning\|Loading"
  python explore_native.py --model $m --sets illusynP 2>&1 | grep -v "Warning\|Loading"
  G3_DIR=results/g7_syn G3_SETS=illusynP python diag_gen.py --model $m 2>&1 | grep -v "Warning\|Loading"
done
echo G7DONE
