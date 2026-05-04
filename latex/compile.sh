#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Compile the time-varying-parameter ES-MDA report.
#
# Usage:
#   ./compile.sh              # build report.pdf with latexmk (default)
#   ./compile.sh clean        # remove auxiliary files but keep PDF
#   ./compile.sh distclean    # remove auxiliary files AND PDF
#   ./compile.sh watch        # rebuild on file changes (requires latexmk)
#
# Requires a working TeX distribution (TeX Live, MacTeX, MiKTeX) with
# `latexmk` available on PATH.  Falls back to a manual pdflatex+bibtex
# loop if latexmk is not installed.
# ----------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")"

DOC=report
ACTION=${1:-build}

clean_aux() {
  rm -f \
    "$DOC".aux "$DOC".bbl "$DOC".bcf "$DOC".blg "$DOC".fdb_latexmk \
    "$DOC".fls "$DOC".log "$DOC".out "$DOC".run.xml "$DOC".synctex.gz \
    "$DOC".toc "$DOC".lof "$DOC".lot "$DOC".nav "$DOC".snm "$DOC".vrb \
    sections/*.aux
}

build_with_latexmk() {
  latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error "$DOC".tex
}

build_manual() {
  pdflatex -interaction=nonstopmode -halt-on-error "$DOC".tex
  bibtex "$DOC" || true
  pdflatex -interaction=nonstopmode -halt-on-error "$DOC".tex
  pdflatex -interaction=nonstopmode -halt-on-error "$DOC".tex
}

case "$ACTION" in
  build)
    if command -v latexmk >/dev/null 2>&1; then
      build_with_latexmk
    else
      echo "latexmk not found -- falling back to manual pdflatex/bibtex loop." >&2
      build_manual
    fi
    echo "Built $DOC.pdf"
    ;;
  watch)
    if ! command -v latexmk >/dev/null 2>&1; then
      echo "watch mode requires latexmk on PATH" >&2
      exit 1
    fi
    latexmk -pdf -bibtex -pvc -interaction=nonstopmode "$DOC".tex
    ;;
  clean)
    clean_aux
    echo "Removed auxiliary files (kept $DOC.pdf)."
    ;;
  distclean)
    clean_aux
    rm -f "$DOC".pdf
    echo "Removed auxiliary files and $DOC.pdf."
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    echo "Usage: $0 [build|watch|clean|distclean]" >&2
    exit 2
    ;;
esac
