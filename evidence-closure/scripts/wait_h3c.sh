#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  grep -q Traceback results/h3_dev2_*.log && { echo H3C_FAILED; tail -n 5 results/h3_dev2_*.log; exit 1; }
  a=$(cat results/h3_dev2_qwen25vl7b.jsonl | wc -l); b=$(cat results/h3_dev2_qwen3vl8b.jsonl | wc -l)
  [ "$a" -ge 630 ] && [ "$b" -ge 630 ] && break
  sleep 300
done
for t in qwen25vl7b qwen3vl8b; do .venv/bin/python -W ignore scripts/h3c_policy.py --tag $t --dev2 > results/h3c_${t}_dev2.log 2>&1 & done
wait; echo H3C_DONE
