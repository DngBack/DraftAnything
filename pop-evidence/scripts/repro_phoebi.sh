#!/usr/bin/env bash
# Stage A: reproduce PHOEBI's decoders without Slurm (plan §10). Runs inside the PHOEBI clone with
# this project's venv; logs commit, GPU, torch and every command to reports/experiment_log.md.
#   bash scripts/repro_phoebi.sh fetch     # needs `hf auth login` + accepted terms on the dataset card
#   bash scripts/repro_phoebi.sh random    # three decoders, random split
#   bash scripts/repro_phoebi.sh lco       # LCO seeds 1337/1338/1339 (= slurm/run_lco*.sh) + aggregation
#   bash scripts/repro_phoebi.sh devcv     # closed-form decoders on the 5 dev folds (comparable to our pooled dev)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HERE/.venv/bin/python"
PHOEBI_ROOT="${PHOEBI_ROOT:-/home/jovyan/bachdx2/datasets/phoebi/repo}"
LOG="$HERE/reports/experiment_log.md"
cd "$PHOEBI_ROOT"

run() {
  echo "- $(date '+%F %T') \`$*\` (phoebi@$(git rev-parse --short HEAD))" >> "$LOG"
  local t0=$SECONDS
  "$@"
  echo "  - done in $((SECONDS - t0)) s" >> "$LOG"
}

{
  echo ""; echo "## repro_phoebi $1 — $(date '+%F %T')"
  echo "- GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "- torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
} >> "$LOG"

case "$1" in
  fetch)  run $PY tools/fetch_dataset.py ;;
  random)
    run $PY -m src.simplex_unmixing.train --epochs 30
    run $PY -m src.prototype_matching.train
    run $PY -m src.mc_channel.train
    run $PY experiments/run_presence_detection.py --method simplex --model_dir outputs/simplex_unmixing/default
    run $PY experiments/run_presence_detection.py --method prototype --model_dir outputs/prototype_matching/default
    run $PY -m src.mc_channel.test_eval --model_dir outputs/mc_channel/default ;;
  lco)
    # = slurm/run_lco.sh (seed 1337) + slurm/run_lco_extra_seeds.sh (1338, 1339); check OUTDIR naming there
    run $PY experiments/run_phoebi_heldout.py --output_dir outputs/phoebi_heldout
    for s in 1338 1339; do run $PY experiments/run_phoebi_heldout.py --seed $s --output_dir outputs/phoebi_heldout_seed$s; done
    run $PY experiments/aggregate_seeds.py ;;
  devcv)
    # nested CV reads the Stage 0 unit cache, so Stage 0 runs first (GPU, one pass per tile size)
    run $PY experiments/run_comp_stage0_granularity.py --output_dir outputs/compositional/stage0 --granularities cls224 patchmean224
    run $PY experiments/run_comp_nested_cv.py --granularities cls224 patchmean224 ;;
  *) echo "usage: $0 fetch|random|lco|devcv"; exit 1 ;;
esac
