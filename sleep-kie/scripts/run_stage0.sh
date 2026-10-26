#!/usr/bin/env bash
# Locked stage-0 matrix (docs/stage0-protocol.md): 7 variants x 3 seeds, round-robin over GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-6000}
GPUS=(${GPUS:-0 1})
JOBS=${JOBS:-4}          # concurrent runs per GPU
mkdir -p results/stage0
i=0
for seed in 0 1 2; do
  for m in A0_N1 A1_N2 A2_N4 A3_N4_half A4_N1_2x A5_deep4 full; do
    [ -f results/stage0/$m-s$seed/metrics.json ] && continue
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py --model $m --seed $seed --steps $STEPS --tag stage0 \
      > results/stage0/$m-s$seed.out 2>&1 &
    i=$((i + 1))
    while [ "$(jobs -rp | wc -l)" -ge $((JOBS * ${#GPUS[@]})) ]; do wait -n; done
  done
done
wait
python scripts/report.py --tag stage0
