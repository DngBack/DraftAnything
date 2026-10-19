#!/bin/bash
# G5 C queue (docs/g5_protocol.md): ICL on 4 more models, k = 0 baselines, format variants on Qwen3-VL-8B; GPU 0 only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python explore_icl.py --model qwen3vl8b --ks 0 4 8 --fmt plain 2>&1 | grep "ICL"
for f in instr alt single; do python explore_icl.py --model qwen3vl8b --ks 4 8 --fmt $f 2>&1 | grep "ICL\|Error"; done
python explore_icl.py --model qwen35_9b --ks 0 2>&1 | grep "ICL\|Error"
for m in qwen3vl4b gemma3_4b internvl35_8b qwen35_4b; do python explore_icl.py --model $m --ks 0 1 2 4 8 2>&1 | grep "ICL\|Error"; done
echo ICL5DONE
