#!/bin/bash
# R2 F: run the remaining grid models once a GPU has enough free memory (shared node). usage: ./run_r2_when_free.sh <gpu> <MiB> <g3_dir> model...
gpu=$1; need=$2; shift 2
until [ "$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $need ]; do sleep 120; done
./run_r2.sh $gpu grid "$@"
