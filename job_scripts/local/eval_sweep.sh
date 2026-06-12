#!/bin/bash
# LOCAL (no SLURM) post-processing of a rollout-ESMDA sweep -- the local sibling
# of job_scripts/snellius/eval_sweep.slurm. Runs the last two stages of the
# three-script pipeline (run_esmda.py / the sweep launchers are the first):
#
#   Stage 1  scripts/compute_sweep_metrics.py  -- reads the per-run posterior
#            results in the RUNS folder + each run's ground truth, writes SMALL
#            metric artifacts to METRICS_DIR.
#   Stage 2  scripts/compare_sweep_results.py  -- reads METRICS_DIR, writes the
#            comparison figures + big CSV to COMPARISON_DIR.
#
# The one thing you usually provide is the FOLDER HOLDING ALL THE RUNS (the
# RESULTS_ROOT the sweeps wrote to -- it contains one <model>_nx.._ens.._steps..
# subdir per run). Pass it as the first argument:
#
#     bash job_scripts/local/eval_sweep.sh /path/to/assim_from_ground_truth
#
# With no folder it falls back to $RESULTS_ROOT, then to the repo-local default.
# Anything after the folder is forwarded to the COMPARE stage only (compare-only
# flags like the sweep selector / fixed domain), e.g.:
#
#     bash job_scripts/local/eval_sweep.sh /path/to/runs --sweep ensemble
#     bash job_scripts/local/eval_sweep.sh /path/to/runs --sweep domain --linear-x
#     bash job_scripts/local/eval_sweep.sh --sweep ensemble        # uses default RUNS dir
#
# Restrict BOTH stages to certain backends with the MODELS env var (do NOT pass
# --models positionally -- it would reach compare but crash the compute stage):
#
#     MODELS="pyudales pylbm" bash job_scripts/local/eval_sweep.sh /path/to/runs
#
# Other env knobs: ENV (pixi env, default dev), METRICS_DIR, COMPARISON_DIR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Pixi env carrying the metric/plotting deps (data_assimilation, pyurbanair,
# xarray, matplotlib, ...). "dev" has everything; override with ENV=.
ENV="${ENV:-dev}"

# First positional arg, if it is NOT a flag, is the RUNS folder (the sweep's
# RESULTS_ROOT). Otherwise fall back to $RESULTS_ROOT, then the repo-local default
# the local runners use. Everything left in "$@" goes to the compare stage.
RUNS_DIR_ARG=""
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  RUNS_DIR_ARG="$1"; shift
fi
RUNS_DIR="${RUNS_DIR_ARG:-${RESULTS_ROOT:-${REPO_ROOT}/results/assim_from_ground_truth}}"

# Where the small metric artifacts (stage 1) and the figures + CSV (stage 2) land.
METRICS_DIR="${METRICS_DIR:-${REPO_ROOT}/sweep_metrics}"
COMPARISON_DIR="${COMPARISON_DIR:-${REPO_ROOT}/comparison}"

[ -d "${RUNS_DIR}" ] || {
  echo "error: runs folder not found: ${RUNS_DIR}" >&2
  echo "       pass it as the first argument, or set RESULTS_ROOT." >&2
  exit 1
}
# A valid runs folder holds at least one <run>/run_summary.yaml.
if ! compgen -G "${RUNS_DIR}/*/run_summary.yaml" >/dev/null; then
  echo "error: no <run>/run_summary.yaml under ${RUNS_DIR}" >&2
  echo "       is this the folder the sweeps wrote to (their RESULTS_ROOT)?" >&2
  exit 1
fi

# MODELS (env) restricts BOTH stages; "$@" (positional) goes to compare only.
MODELS_ARG=()
[ -n "${MODELS:-}" ] && MODELS_ARG=(--models ${MODELS})

# Single-process metric computation / plotting; keep BLAS single-threaded.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "LOCAL eval_sweep on $(hostname) -- $(date)"
echo "Runs folder:  ${RUNS_DIR}"
echo "Metrics ->    ${METRICS_DIR}"
echo "Comparison -> ${COMPARISON_DIR}"
[ -n "${MODELS:-}" ] && echo "Restricting to models: ${MODELS}"
[ "$#" -gt 0 ] && echo "Compare args: $*"

echo "=== Stage 1: compute_sweep_metrics -> ${METRICS_DIR} ==="
pixi run -e "${ENV}" -- python -u scripts/compute_sweep_metrics.py \
    --root "${RUNS_DIR}" --out "${METRICS_DIR}" "${MODELS_ARG[@]}"

echo "=== Stage 2: compare_sweep_results -> ${COMPARISON_DIR}/{domain,ensemble} ==="
pixi run -e "${ENV}" -- python -u scripts/compare_sweep_results.py \
    --root "${METRICS_DIR}" --out "${COMPARISON_DIR}" "${MODELS_ARG[@]}" "$@"

echo "Done -- metrics in ${METRICS_DIR}, figures + CSV in ${COMPARISON_DIR}  ($(date))"
