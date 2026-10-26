#!/usr/bin/env bash
# Run "model:seed" jobs one by one, launching each on whichever GPU has >= NEED_MIB free.
# usage: TAG=pilot5 STEPS=12000 EXTRA=--dev bash scripts/queue.sh A0_N1:0 A2_N4:0 ...
set -uo pipefail
cd "$(dirname "$0")/.."
NEED_MIB=${NEED_MIB:-16000}
mkdir -p results/$TAG
for job in "$@"; do
  m=${job%:*}; s=${job#*:}
  [ -f results/$TAG/$m-s$s/metrics.json ] && continue
  while true; do
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
          awk -F', ' -v need=$NEED_MIB '$2 >= need {print $1; exit}')
    [ -n "$gpu" ] && break
    sleep 60
  done
  echo "$(date +%T) start $m s$s on GPU $gpu"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu nohup python scripts/train.py --model $m --seed $s --steps $STEPS --tag $TAG ${EXTRA:-} \
    > results/$TAG/$m-s$s.out 2>&1 &
  sleep 240   # let the job reach peak memory before probing again
done
wait
echo "$(date +%T) all done"
