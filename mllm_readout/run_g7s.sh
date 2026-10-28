#!/bin/bash
# G7 I (exploratory) queue; GPU 0 only; waits for run_g7.sh
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
while pgrep -f run_g7.sh > /dev/null; do sleep 60; done
for m in qwen3vl8b qwen3vl4b internvl35_8b llava15_7b; do
  python -u g7_steer.py --model $m 2>&1 | grep --line-buffered -v "Warning\|Loading"
done
echo G7SDONE
