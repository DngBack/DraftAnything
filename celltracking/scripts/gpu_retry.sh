#!/bin/bash
# Run a command; if it dies from CUDA OOM (shared GPUs), wait and retry.
# usage: scripts/gpu_retry.sh LOG cmd args...
log=$1; shift
for try in $(seq 1 60); do
  "$@" > "$log" 2>&1 && exit 0
  grep -q -i "out of memory\|CUDA_ERROR_OUT_OF_MEMORY\|ALLOC_FAILED" "$log" || exit 1
  echo "retry $try after OOM $(date)" >> "$log.retries"
  sleep 120
done
exit 1
