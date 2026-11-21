#!/usr/bin/env bash
# Wait until both _lt runs have their 100k snapshot, stop them (kept as the v1 ablation), start the v2 tracks.
set -euo pipefail
cd "$(dirname "$0")/.."
for a in char nfd; do until [ -e runs/gen_vn_${a}_lt/snap_100000.pt ]; do echo "[$(date '+%F %T')] waiting for $a snap_100000"; sleep 120; done; done
for a in char nfd; do
  pkill -f "pipeline.sh gen_${a}$" || true          # first the track, so it does not move on to IAM
  sleep 2
  pkill -f "train_generator.py --dataset vn --vn-data data/vn --content $a .*gen_vn_${a}_lt" || true
  echo '{"reason":"v1 stopped at 100k; continued as gen_vn_'$a'_v2 (multiword 0.2, mark weight 3)"}' > runs/gen_vn_${a}_lt/STOPPED
done
sleep 20
for t in gen_char_v2 gen_nfd_v2; do
  tmux new-window -d -t partscribe -n $t -c "$PWD" "bash scripts/pipeline.sh $t; echo '--- track $t exited'; exec bash"
done
tmux new-window -d -t partscribe -n eval_v2 -c "$PWD" \
  "python scripts/eval_loop.py --tag _v2 --ocr runs/eval_ocr_vn_local/best.pt --pairs 2>&1 | tee -a runs/eval_loop_v2.log; exec bash"
echo "[$(date '+%F %T')] switched to v2"
