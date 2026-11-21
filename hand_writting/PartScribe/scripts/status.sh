#!/usr/bin/env bash
# One-screen status of the PartScribe run (used by the tmux "status" window).
cd "$(dirname "$0")/.."
for d in tok_slots_strict tok_global_strict vae_vn32 vae_iam64 gen_vn_char gen_vn_nfd gen_vn_char_lt gen_vn_nfd_lt gen_vn_char_v2 gen_vn_nfd_v2 gen_iam_char gen_iam_nfd gen_vn_parts eval_ocr_vn eval_ocr_vn_local; do
  f=runs/$d/log.jsonl
  if [ -e runs/$d/STOPPED ]; then s=STOPPED
  elif [ -e runs/$d/DONE ] || [ -e runs/$d/final.pt ]; then s=DONE
  elif [ -e "$f" ]; then s=running; else s=pending; fi
  last=$( [ -e "$f" ] && tail -1 "$f" | cut -c1-150 )
  printf '%-18s %-8s %s\n' "$d" "$s" "$last"
done
[ -e runs/probes/verdict.txt ] && { echo; cat runs/probes/verdict.txt; }
for c in runs/eval/vn*_*/compare.json; do [ -e "$c" ] && { echo; echo "== $c (A=char, B=nfd)"; python3 -c "import json,sys;d=json.load(open('$c'));[print(f'{k:40s} A {v[\"A\"]:.3f}  B {v[\"B\"]:.3f}  B-A {v[\"diff_B_minus_A\"]:+.3f} CI {v[\"ci95\"][0]:+.3f}..{v[\"ci95\"][1]:+.3f}') for k,v in d.items() if isinstance(v,dict)]"; }; done
echo
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
