#!/usr/bin/env bash
# All variants x 3 seeds, PAR jobs at a time on one GPU (models are small; features are cached).
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-0}; PAR=${PAR:-6}; TAG=${TAG:-main}
VARIANTS=${VARIANTS:-"point pool set2 set4 cem_r1 cem_r2 cem_r4 cem_r8 cem_r4_nores"}
mkdir -p logs
for s in 0 1 2; do
  for v in $VARIANTS; do
    [ -f "results/$TAG/$v-s$s/metrics.json" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python scripts/train.py --model "$v" --seed "$s" --tag "$TAG" \
      > "logs/$TAG-$v-s$s.out" 2>&1 &
  done
done
wait
