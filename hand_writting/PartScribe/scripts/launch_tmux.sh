#!/usr/bin/env bash
# Start the PartScribe run in tmux session "partscribe" (one window per track + status).
#   tmux attach -t partscribe        (windows: status, stage1, vae_vn, vae_iam, gen_char, gen_parts)
#        # Ctrl-b n / Ctrl-b p to switch windows, Ctrl-b d to detach
set -euo pipefail
cd "$(dirname "$0")/.."
S=partscribe
tmux has-session -t $S 2>/dev/null && { echo "session $S already exists"; exit 1; }
tmux new-session -d -s $S -n status -c "$PWD" "watch -n 60 bash scripts/status.sh"
for track in stage1 vae_vn vae_iam gen_char gen_parts; do
  tmux new-window -t $S -n $track -c "$PWD" "bash scripts/pipeline.sh $track; echo '--- track $track exited'; exec bash"
done
tmux list-windows -t $S
