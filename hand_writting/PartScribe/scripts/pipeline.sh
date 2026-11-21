#!/usr/bin/env bash
# PartScribe real run, one track per tmux window (see scripts/launch_tmux.sh).
#
#   bash scripts/pipeline.sh stage1    # tokenizers -> probes -> content table
#   bash scripts/pipeline.sh vae_vn    # VAE 32 px (Vietnamese)
#   bash scripts/pipeline.sh vae_iam   # VAE 64 px (IAM)
#   bash scripts/pipeline.sh gen_char  # VN char baseline, then IAM char baseline
#   bash scripts/pipeline.sh gen_parts # VN parts, then IAM parts (stopped after stage-1 NO-GO)
#   bash scripts/pipeline.sh eval_ocr  # independent evaluator OCR (full train + HAT, never val/test)
#   bash scripts/pipeline.sh gen_nfd   # VN nfd, then IAM nfd (main arm, docs/stage1_deviation.md)
#
# Long tail: generators use VATr-pp's ComponentCorpusSampler weights on target images
# (--rare-fraction 0.3, component mix 0.5). VN runs: gen_vn_<mode>_lt.
# gen_vn_char / gen_vn_nfd (natural frequency, stopped at 25k) are kept as the sampler ablation.
#
# v2 (tracks gen_char_v2 / gen_nfd_v2): continue gen_vn_<mode>_lt from its 100k snapshot to
# 200k with --multiword 0.2 (same-writer phrases: the data has almost no spaces) and
# --mark-weight 3 (flow loss up-weighted on diacritics/dots). IAM runs use the same options.
# Joined words are put on one baseline (tall canvas, then scaled to h); the first v2 runs with
# unaligned joins were stopped at ~110k and kept as gen_vn_<mode>_v2a.
# VN v2 also appends 16 HAT2023 words per batch as style-free content targets (VATr-pp's HAT
# task; capitals: 21k all-caps words). HAT has no writer ids, so style is dropped for them;
# HAT words with a held-out character in either case are removed.
#
# Data policy follows VATr-pp (merged VNOnDB+ViHTGen, 265 train / 30 dev writers, source-balanced
# sampling, 15 supports at 6 x height) plus holdout protocol v1.
# Every step writes a marker; re-running a track skips finished steps
# (generators resume from last.pt).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
VN=data/vn
GEN_STEPS=${GEN_STEPS:-200000}        # VN: batch 16 writers x 4
IAM_STEPS=${IAM_STEPS:-100000}        # IAM 64 px: batch 8 x 4, accum 2 (effective 64; memory on the shared GPU)
PY=python

log() { echo "[$(date '+%F %T')] $*"; }
wait_for() { until [ -e "$1" ]; do log "waiting for $1"; sleep 300; done; }
run() {  # run <out_dir> <cmd...> ; skip when <out_dir>/DONE exists
  local out=$1; shift
  if [ -e "$out/DONE" ]; then log "skip $out (done)"; return; fi
  mkdir -p "$out"
  log "start $out: $*"
  "$@" 2>&1 | tee -a "$out/stdout.log"
  log "end $out"
}

case "${1:?track}" in
stage1)
  run runs/tok_slots_strict  $PY scripts/train_tokenizer.py --model slots  --variant strict --out runs/tok_slots_strict
  run runs/tok_global_strict $PY scripts/train_tokenizer.py --model global --variant strict --out runs/tok_global_strict
  mkdir -p runs/probes
  [ -e runs/probes/pixels.json ] || $PY scripts/run_probes.py --pixels --out runs/probes/pixels.json | tail -3
  [ -e runs/probes/slots.json ]  || $PY scripts/run_probes.py --ckpt runs/tok_slots_strict/best.pt  --out runs/probes/slots.json | tail -3
  [ -e runs/probes/global.json ] || $PY scripts/run_probes.py --ckpt runs/tok_global_strict/best.pt --out runs/probes/global.json | tail -3
  $PY scripts/check_stage1.py | tee runs/probes/verdict.txt
  [ -e data/content_strict.pt ] || $PY scripts/build_content_table.py --ckpt runs/tok_slots_strict/best.pt --out data/content_strict.pt
  log "stage1 finished"
  ;;
vae_vn)
  [ -e runs/vae_vn32/final.pt ] || { mkdir -p runs/vae_vn32; $PY scripts/train_vae.py --dataset vn --vn-data $VN --out runs/vae_vn32 2>&1 | tee -a runs/vae_vn32/stdout.log; }
  ;;
vae_iam)
  [ -e runs/vae_iam64/final.pt ] || { mkdir -p runs/vae_iam64; $PY scripts/train_vae.py --dataset iam --out runs/vae_iam64 2>&1 | tee -a runs/vae_iam64/stdout.log; }
  ;;
gen_char|gen_parts|gen_nfd)
  mode=${1#gen_}
  extra=()
  if [ "$mode" = parts ]; then wait_for data/content_strict.pt; extra=(--table data/content_strict.pt); fi
  wait_for runs/vae_vn32/final.pt
  run runs/gen_vn_${mode}_lt $PY scripts/train_generator.py --dataset vn --vn-data $VN --content $mode "${extra[@]}" \
      --vae runs/vae_vn32/final.pt --out runs/gen_vn_${mode}_lt --steps $GEN_STEPS --rare-fraction 0.3 --resume
  wait_for runs/vae_iam64/final.pt
  run runs/gen_iam_$mode $PY scripts/train_generator.py --dataset iam --content $mode "${extra[@]}" \
      --vae runs/vae_iam64/final.pt --out runs/gen_iam_$mode --steps $IAM_STEPS --writers-per-batch 8 --accum 2 --rare-fraction 0.3 --resume
  ;;
gen_char_v2|gen_nfd_v2)
  mode=${1#gen_}; mode=${mode%_v2}
  V2=(--multiword 0.2 --mark-weight 3 --rare-fraction 0.3)
  wait_for runs/gen_vn_${mode}_lt/snap_100000.pt
  run runs/gen_vn_${mode}_v2 $PY scripts/train_generator.py --dataset vn --vn-data $VN --content $mode \
      --vae runs/vae_vn32/final.pt --out runs/gen_vn_${mode}_v2 --steps $GEN_STEPS "${V2[@]}" \
      --content-only-per-batch 16 --init-from runs/gen_vn_${mode}_lt/snap_100000.pt --resume
  wait_for runs/vae_iam64/final.pt
  run runs/gen_iam_$mode $PY scripts/train_generator.py --dataset iam --content $mode \
      --vae runs/vae_iam64/final.pt --out runs/gen_iam_$mode --steps $IAM_STEPS --writers-per-batch 8 --accum 2 \
      "${V2[@]}" --resume
  ;;
gen_char_v2_300k|gen_nfd_v2_300k)
  # 2026-11-20: v2 finished 200k with the flow loss still falling ~0.011 per 20k steps and FID/KID
  # still improving, so both arms continue from their own last.pt (step 200k, constant LR, same
  # options) to 300k. --seed 1 so the data order does not replay the first 100k v2 batches.
  # No IAM afterwards (IAM is on hold). The 200k DONE marker is kept as DONE_200k.
  mode=${1#gen_}; mode=${mode%_v2_300k}
  out=runs/gen_vn_${mode}_v2
  [ -e $out/DONE ] && [ "$(cat $out/DONE)" = 200000 ] && mv $out/DONE $out/DONE_200k
  run $out $PY scripts/train_generator.py --dataset vn --vn-data $VN --content $mode \
      --vae runs/vae_vn32/final.pt --out $out --steps 300000 --multiword 0.2 --mark-weight 3 --rare-fraction 0.3 \
      --content-only-per-batch 16 --seed 1 --resume
  ;;
eval_ocr)
  run runs/eval_ocr_vn $PY scripts/train_evaluator.py --vn-data $VN --out runs/eval_ocr_vn
  ;;
*) echo "unknown track $1"; exit 1 ;;
esac
