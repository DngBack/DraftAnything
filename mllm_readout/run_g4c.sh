#!/bin/bash
# G4 follow-ups, GPU0: matched-capacity control (MLLM image-token grid at the steering depth), Qwen3-VL-4B R2 with idx.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
P=.venv/bin/python
$P explore_input.py --deep --src qwen3vl4b qwen3vl8b qwen35_2b qwen35_4b qwen35_9b gemma3_4b qwen3vl2b qwen2vl_7b llava15_7b --sets illusions countbench
for s in illusions countbench mmvp; do [ -f results/explore_hcfg/qwen3vl4b_$s.npz ] || $P explore_hcfg.py --model qwen3vl4b --set $s; done
$P explore_input.py --deep --src gemma3_27b qwen3vl32b --sets illusions countbench
echo G4CDONE
