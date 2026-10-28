#!/bin/bash
# G4 repair matrix (docs/g4_protocol.md), GPU0. usage: ./run_g4.sh A|B   (A: steering, R5, R2; B: R4 thinking)
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
P=.venv/bin/python; Q35="qwen35_2b qwen35_4b qwen35_9b"; QVL="qwen3vl4b qwen3vl8b"; S="illusions countbench mmvp"
mkdir -p logs results/g4
if [ "$1" = A ]; then
  for m in $Q35; do G3_DIR=results/g3_heldout G3_SETS=mmvp $P diag_gen.py --model $m; done
  for m in $Q35; do G3_DIR=results/g3_heldout STEER_SETS=mmvp STEER_ALPHA=16 $P steer_g3.py --model $m; done
  for m in $QVL; do STEER_SETS=mmvp STEER_ALPHA=16 $P steer_g3.py --model $m; done
  for m in $Q35; do G3_DIR=results/g3_heldout STEER_SRC=ext STEER_ALPHA=16 $P steer_g3.py --model $m; done
  for m in $QVL; do STEER_SRC=ext STEER_ALPHA=16 $P steer_g3.py --model $m; done
  for m in $QVL $Q35; do $P explore_repair.py --model $m --mode res2 --sets $S; done
  for m in qwen3vl8b $Q35; do for s in $S; do
    [ -f results/explore_hcfg/${m}_$s.npz ] || $P explore_hcfg.py --model $m --set $s; done; done
  echo G4ADONE
else
  for m in $Q35; do $P explore_repair.py --model $m --mode think --sets $S; done
  echo G4BDONE
fi
