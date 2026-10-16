#!/bin/bash
# R2 (docs/r2_protocol.md). usage: ./run_r2.sh <gpu> grid <g3_dir> model...   |   ./run_r2.sh <gpu> whatsup model...
gpu=$1; mode=$2; shift 2
export CUDA_VISIBLE_DEVICES=$gpu
mkdir -p logs
if [ $mode = grid ]; then
  export G3_DIR=$1 STEER_GRID=0,1,2,4,8,16,32,64; shift
  for m in "$@"; do .venv/bin/python steer_g3.py --model $m >> logs/r2_grid_$m.log 2>&1; echo "exit $? $(date +%H:%M)" >> logs/r2_grid_$m.log; done
else
  for m in "$@"; do .venv/bin/python pilot.py --model $m --family spatial_real --n 820 --states_only >> logs/r2_whatsup_$m.log 2>&1; done
fi
