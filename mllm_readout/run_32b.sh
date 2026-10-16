#!/bin/bash
# Scale point Qwen3-VL-32B: waits for the download and for GPU0's Gemma-27B queue, then runs every analysis on GPU0.
cd "$(dirname "$0")"
until grep -q DOWNLOADED logs/dl_qwen32b.log && grep -q ALLDONE logs/gencal_27b.log; do sleep 60; done
./run_g15.sh 0 "qwen3vl32b count" "qwen3vl32b spatial" "qwen3vl32b count_real" "qwen3vl32b spatial_real" "qwen3vl32b color_cf" "qwen3vl32b attr" | grep -v ALLDONE
for t in countbench vlmbias; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python gen_calib.py --task $t --model qwen3vl32b; done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python cf_repro.py --model qwen3vl32b
CUDA_VISIBLE_DEVICES=0 .venv/bin/python count_repro.py --model qwen3vl32b
echo ALLDONE
