#!/bin/bash
# probe-in-disguise study (exploratory, 2026-10-18), GPU0. usage: ./run_disguise.sh 1|2
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 STEER_ALPHA=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
if [ "$1" = 1 ]; then
  for m in gemma3_4b internvl35_8b llava15_7b qwen2vl_7b qwen3vl2b gemma3_27b qwen3vl32b; do STEER_SRC=ext $P steer_g3.py --model $m; done
  for m in qwen35_4b; do for s in oracle anti; do G3_DIR=results/g3_heldout STEER_SRC=$s $P steer_g3.py --model $m; done; done
else
  for m in qwen3vl4b gemma3_4b internvl35_8b llava15_7b qwen2vl_7b; do for s in oracle anti; do STEER_SRC=$s $P steer_g3.py --model $m; done; done
fi
echo DISGUISE$1DONE
