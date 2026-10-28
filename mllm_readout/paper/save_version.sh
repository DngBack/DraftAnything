#!/bin/bash
# Compile main and archive as versions/SeeingNotSaying_ver<N>_<YYYYMMDD-HHMM>.pdf (main + supplement appended).
# Usage: paper/save_version.sh "short note"   (VER=<N> overrides the automatic version number)
set -e
case "${1:-}" in -*|"") echo "usage: $0 \"short note\"" >&2; exit 1;; esac
cd "$(dirname "$0")"
for f in main supp; do
  pdflatex -interaction=nonstopmode $f >/dev/null; bibtex $f >/dev/null || true
  pdflatex -interaction=nonstopmode $f >/dev/null; pdflatex -interaction=nonstopmode $f >/dev/null
done
n=${VER:-$(( $(ls versions/SeeingNotSaying_ver*_*.pdf 2>/dev/null | sed -E 's/.*_ver([0-9]+)_.*/\1/' | sort -n | tail -1 || echo 0) + 1 ))}
ts=$(date -u +%Y%m%d-%H%M)
out=versions/SeeingNotSaying_ver${n}_${ts}.pdf
gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile="$out" main.pdf supp.pdf
pages=$(grep -oP "Output written on main.pdf \(\K[0-9]+" main.log)
spages=$(grep -oP "Output written on supp.pdf \(\K[0-9]+" supp.log)
echo "ver${n} ${ts} ${pages}p+${spages}p supp: ${1:-}" >> versions/CHANGELOG.txt
echo "$out"
