#!/bin/bash
# Compile main + supp and archive as versions/LineageRAFT_ver<N>_<YYYYMMDD-HHMM>.pdf (main + supplement appended).
# Usage: paper/save_version.sh "short note"
set -e
case "${1:-}" in -*|"") echo "usage: $0 \"short note\"" >&2; exit 1;; esac
cd "$(dirname "$0")"
for f in main supp; do
  pdflatex -interaction=nonstopmode $f >/dev/null; bibtex $f >/dev/null || true
  pdflatex -interaction=nonstopmode $f >/dev/null; pdflatex -interaction=nonstopmode $f >/dev/null
done
n=$(( $(ls versions/LineageRAFT_ver*_*.pdf 2>/dev/null | sed -E 's/.*_ver([0-9]+)_.*/\1/' | sort -n | tail -1 || echo 0) + 1 ))
ts=$(date -u +%Y%m%d-%H%M)
out=versions/LineageRAFT_ver${n}_${ts}.pdf
pdfunite main.pdf supp.pdf "$out"
echo "ver${n} ${ts} $(pdfinfo main.pdf | awk '/Pages/{print $2}')p+supp: ${1:-}" >> versions/CHANGELOG.txt
echo "$out"
