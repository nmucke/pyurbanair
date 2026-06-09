#!/bin/bash
# Shared LOCAL sweep engine for rollout-ESMDA-from-truth. ONE implementation,
# used by every backend, so all three models sweep the IDENTICAL set of values
# and run the exact same experiment at each point -- only the assimilation solver
# differs.
#
# Not run directly: each backend folder carries three thin wrappers
# (sweep_{domain,ensemble,esmda_steps}_rollout_esmda_from_truth.sh) that call
# this with the sweep kind and the sibling runner:
#
#     bash sweep_base.sh <domain|ensemble|steps> <path/to/rollout_esmda_from_truth.sh> [hydra overrides...]
#
# It runs the backend runner once per swept value, SEQUENTIALLY in this shell
# (local: no SLURM, no partitions, no wall clock). Each runner invocation gets the
# swept value injected via the environment (NX/NY/NZ, ENSEMBLE_SIZE or
# NUM_ESMDA_STEPS) so it lands in its own RESULTS_DIR. Any extra arguments are
# forwarded as Hydra overrides to EVERY run.
#
# A single failing point does NOT abort the sweep; failures are collected and
# reported at the end (exit status 1 if any run failed).
set -uo pipefail

KIND="${1:?usage: sweep_base.sh <domain|ensemble|steps> <runner.sh> [hydra overrides...]}"
RUNNER="${2:?usage: sweep_base.sh <domain|ensemble|steps> <runner.sh> [hydra overrides...]}"
shift 2
[ -f "${RUNNER}" ] || { echo "error: runner '${RUNNER}' not found" >&2; exit 1; }
MODEL="$(basename "$(dirname "${RUNNER}")")"

# ============================================================================
# Canonical sweep value lists -- defined ONCE here so every backend sweeps the
# identical set. Edit these to retune the whole local suite at once.
# ============================================================================
# Resolution sweep (domain): "NX NY NZ" rows. The ground truth is 100 x 80 x 32
# (aspect ratio 25:20:8); each row keeps that ratio, coarse -> ground-truth grid.
RESOLUTIONS=(
  "25 20 8"     # k=1  (coarsest)
  "50 40 16"    # k=2
  # "75 60 24"    # k=3
  # "100 80 32"   # k=4  (== ground-truth resolution)
)
# Ensemble-size sweep, at the fixed grid below.
ENSEMBLE_SIZES=( 8 16 32 64 96 )
# ESMDA-steps sweep (iterations per window), at the fixed grid + ensemble below.
ESMDA_STEPS=( 1 2 3 4 )

# Fixed values for the dimensions a given sweep holds constant.
FIXED_NX="${FIXED_NX:-75}"
FIXED_NY="${FIXED_NY:-60}"
FIXED_NZ="${FIXED_NZ:-24}"
FIXED_ENSEMBLE_SIZE="${FIXED_ENSEMBLE_SIZE:-96}"
FIXED_NUM_ESMDA_STEPS="${FIXED_NUM_ESMDA_STEPS:-3}"
# ============================================================================

FAILURES=()

# Run the backend runner once with the swept dims exported. Extra Hydra overrides
# ("$@" of this script) are forwarded to every run.
run_one() {
  local label="$1"; shift   # remaining args: KEY=VALUE env assignments to export
  echo
  echo "==> [${MODEL}] ${label}"
  if env "$@" bash "${RUNNER}" "${FORWARD[@]}"; then
    echo "==> [${MODEL}] ${label} -- done"
  else
    echo "==> [${MODEL}] ${label} -- FAILED (continuing)" >&2
    FAILURES+=( "${label}" )
  fi
}

# Hydra overrides forwarded to every run.
FORWARD=( "$@" )

case "${KIND}" in
  domain)
    echo "LOCAL [${MODEL}] DOMAIN sweep -- ${#RESOLUTIONS[@]} resolutions (ensemble=${FIXED_ENSEMBLE_SIZE}, steps=${FIXED_NUM_ESMDA_STEPS})."
    [ "${#FORWARD[@]}" -gt 0 ] && echo "Forwarding to every run: ${FORWARD[*]}"
    for res in "${RESOLUTIONS[@]}"; do
      read -r nx ny nz <<<"${res}"
      run_one "NX=${nx} NY=${ny} NZ=${nz}" \
        "NX=${nx}" "NY=${ny}" "NZ=${nz}" \
        "ENSEMBLE_SIZE=${FIXED_ENSEMBLE_SIZE}" "NUM_ESMDA_STEPS=${FIXED_NUM_ESMDA_STEPS}"
    done
    ;;
  ensemble)
    echo "LOCAL [${MODEL}] ENSEMBLE sweep -- ${#ENSEMBLE_SIZES[@]} sizes (grid=${FIXED_NX}x${FIXED_NY}x${FIXED_NZ}, steps=${FIXED_NUM_ESMDA_STEPS})."
    [ "${#FORWARD[@]}" -gt 0 ] && echo "Forwarding to every run: ${FORWARD[*]}"
    for ens in "${ENSEMBLE_SIZES[@]}"; do
      run_one "ENSEMBLE_SIZE=${ens}" \
        "NX=${FIXED_NX}" "NY=${FIXED_NY}" "NZ=${FIXED_NZ}" \
        "ENSEMBLE_SIZE=${ens}" "NUM_ESMDA_STEPS=${FIXED_NUM_ESMDA_STEPS}"
    done
    ;;
  steps)
    echo "LOCAL [${MODEL}] ESMDA-STEPS sweep -- ${#ESMDA_STEPS[@]} step counts (grid=${FIXED_NX}x${FIXED_NY}x${FIXED_NZ}, ensemble=${FIXED_ENSEMBLE_SIZE})."
    [ "${#FORWARD[@]}" -gt 0 ] && echo "Forwarding to every run: ${FORWARD[*]}"
    for steps in "${ESMDA_STEPS[@]}"; do
      run_one "NUM_ESMDA_STEPS=${steps}" \
        "NX=${FIXED_NX}" "NY=${FIXED_NY}" "NZ=${FIXED_NZ}" \
        "ENSEMBLE_SIZE=${FIXED_ENSEMBLE_SIZE}" "NUM_ESMDA_STEPS=${steps}"
    done
    ;;
  *)
    echo "error: unknown sweep kind '${KIND}' (expected domain|ensemble|steps)" >&2
    exit 1
    ;;
esac

echo
if [ "${#FAILURES[@]}" -eq 0 ]; then
  echo "[${MODEL}] ${KIND} sweep complete -- all runs succeeded."
else
  echo "[${MODEL}] ${KIND} sweep finished with ${#FAILURES[@]} failure(s): ${FAILURES[*]}" >&2
  exit 1
fi
