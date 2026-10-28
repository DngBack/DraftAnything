#!/bin/bash
# exploratory input-side features (explore_input.py); GPU0
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
S="illusions countbench carlogos dice tally"
.venv/bin/python explore_input.py --src siglip2 dinov2 --sets $S mmvp >> logs/explore_input.log 2>&1
.venv/bin/python explore_input.py --src qwen3vl4b qwen3vl8b gemma3_4b qwen3vl2b qwen2vl_7b llava15_7b --sets $S >> logs/explore_input.log 2>&1
.venv/bin/python explore_input.py --src qwen35_2b qwen35_4b qwen35_9b qwen2vl_2b --sets illusions countbench >> logs/explore_input.log 2>&1
.venv/bin/python explore_input.py --src gemma3_27b qwen3vl32b --sets $S >> logs/explore_input.log 2>&1
echo INPUTDONE >> logs/explore_input.log
