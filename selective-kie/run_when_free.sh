#!/bin/bash
# Resume run_vlm.py per split whenever a GPU has >=30GB free (GPUs are shared). Needs: 100 val, 100 test, 800 train.
cd "$(dirname "$0")"
declare -A N=([test]=100 [train]=800)
while true; do
  left=0
  for s in test train; do
    n=$(cat outputs/Qwen3-VL-4B-Instruct/$s.jsonl 2>/dev/null | wc -l)
    [ "$n" -ge "${N[$s]}" ] && continue
    left=1
    pgrep -f "run_vlm.py $s" >/dev/null && continue
    for g in 0 1; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      if [ "$free" -ge 30000 ] && ! pgrep -f "GPU$g-claimed" >/dev/null; then
        echo "$(date +%T) start $s on GPU$g (free ${free}MiB, have $n)"
        (exec -a "GPU$g-claimed" bash -c "CUDA_VISIBLE_DEVICES=$g .venv/bin/python run_vlm.py $s >> logs/run_$s.log 2>&1") &
        sleep 120; break
      fi
    done
  done
  [ $left = 0 ] && { echo "all splits done"; exit 0; }
  sleep 60
done
