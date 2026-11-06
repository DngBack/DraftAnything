#!/bin/bash
# Protocol D main run. usage: chain_decide.sh GPU MODEL_DIR TAG   (2 shards, each retried if it dies; output is resumable)
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu=$1; model=$2; tag=$3
lane() {
  for try in 1 2 3 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -W ignore scripts/run_decide.py --model "$model" --tag "$tag" --shard "$1/2" >> "results/decide_${tag}_$1.log" 2>&1 && return 0
    sleep 300
  done
  return 1
}
lane 0 & lane 1 &
wait
