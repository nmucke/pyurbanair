#!/usr/bin/env bash
#==============================================================================
#  compile.sh -- build the manuscript
#
#  Usage:
#    ./compile.sh          build main.pdf
#    ./compile.sh clean     remove LaTeX aux files (keeps the PDF)
#    ./compile.sh cleanall  remove aux files and the PDF
#==============================================================================
set -euo pipefail

cd "$(dirname "$0")"

MAIN="main"

aux_files=(
  "$MAIN".{aux,bbl,blg,log,out,toc,lof,lot,fls,fdb_latexmk,spl,synctex.gz}
  sections/*.aux
)

case "${1:-build}" in
  clean)
    rm -f "${aux_files[@]}"
    echo "Removed auxiliary files."
    exit 0
    ;;
  cleanall)
    rm -f "${aux_files[@]}" "$MAIN".pdf
    echo "Removed auxiliary files and $MAIN.pdf."
    exit 0
    ;;
  build) ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Usage: ./compile.sh [clean|cleanall]" >&2
    exit 1
    ;;
esac

# Prefer latexmk (handles the pdflatex/bibtex/pdflatex/pdflatex loop itself).
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error "$MAIN".tex
else
  echo "latexmk not found; falling back to manual pdflatex + bibtex passes."
  pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
  bibtex "$MAIN"
  pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
  pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex
fi

echo
echo "Done: $MAIN.pdf"
