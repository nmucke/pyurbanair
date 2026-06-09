#!/bin/bash
# LOCAL (no SLURM) rollout-ESMDA-from-truth runner -- pyudales backend (CPU).
#
# Local sibling of job_scripts/snellius/pyudales/rollout_esmda_from_truth.slurm:
# it runs scripts/run_esmda.py DIRECTLY (no sbatch / module / SLURM env vars),
# keeping all heavy I/O and outputs under the repo (pyurbanair). Run config that
# is shared with the pylbm/pypalm runners lives in ../common.sh (sourced below);
# only the pyudales/CPU specifics are set here.
#
# pyudales is CPU-only: the ensemble fans out across parallel processes
# (ensemble.num_parallel_processes = min(ENSEMBLE_SIZE, LOCAL_MAX_PARALLEL), the
# max you choose -- default 16, see common.sh).
#
# Run it from anywhere (it cd's to the repo root itself):
#
#     bash job_scripts/local/pyudales/rollout_esmda_from_truth.sh
#
# Extra Hydra overrides may be appended and take precedence, e.g.:
#
#     bash job_scripts/local/pyudales/rollout_esmda_from_truth.sh esmda.num_steps=4
#
# NX/NY/NZ, ENSEMBLE_SIZE and NUM_ESMDA_STEPS are read from the environment (the
# sweep launchers in this folder set them), so each configuration lands in its
# own RESULTS_DIR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Backend-specific knobs ------------------------------------------------------
ENV="${ENV:-dev}"              # pyudales is CPU-only; "dev" carries every solver feature
ASSIM_MODEL="pyudales"

# Sweep parameters (grid resolution / ensemble / ESMDA steps). Env-overridable so
# the sweep launchers can inject one value per run; each lands in its own RESULTS_DIR.
NX="${NX:-50}"
NY="${NY:-40}"
NZ="${NZ:-16}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-96}"
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-20.0}"   # obs.interval_seconds: time-aggregation bin width [s]
NUM_PARALLEL="${NUM_PARALLEL:-}"    # empty -> min(ENSEMBLE_SIZE, LOCAL_MAX_PARALLEL)

# Shared defaults: paths, domain bounds + sensors, windows, time horizon, dynamic
# parameter settings, localization, ground-truth resolution/validation, and the
# COMMON_RUN_FLAGS array of every shared Hydra override.
source "${REPO_ROOT}/job_scripts/local/common.sh"

# Worker count: fan the ensemble out across up to LOCAL_MAX_PARALLEL parallel
# processes -- the maximum YOU choose (default 16; set it in common.sh or per run
# via LOCAL_MAX_PARALLEL=…). Capped at the ensemble size so no worker sits idle;
# pin an exact count with NUM_PARALLEL=…. Runs are sequential (no scheduler).
if [ -z "${NUM_PARALLEL}" ]; then
  NUM_PARALLEL=$(( ENSEMBLE_SIZE < LOCAL_MAX_PARALLEL ? ENSEMBLE_SIZE : LOCAL_MAX_PARALLEL ))
fi

RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}_int${INTERVAL_SECONDS}${LOCALIZATION_TAG}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_TAG}"
RUN_TEMP_DIR="${TEMP_ROOT}/${RUN_TAG}_$$"

mkdir -p "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "LOCAL pyudales rollout-ESMDA on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} windows=${NUM_ASSIM_WINDOWS}"
echo "ESMDA steps=${NUM_ESMDA_STEPS} obs_interval=${INTERVAL_SECONDS}s localization=${USE_LOCALIZATION}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

# pyudales scratch lands under this run's private temp dir.
EXTRA_FLAGS=(
  "assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}"
  "+assim_model.forward_model.output_dir=${RUN_TEMP_DIR}/outputs"
)

# COMMON_RUN_FLAGS (from common.sh) carries every shared Hydra override; only the
# assim model, the per-run sweep values, hydra.run.dir and the pyudales solver
# flags are added here.
pixi run -e "${ENV}" -- python -u \
    scripts/run_esmda.py \
    "${COMMON_RUN_FLAGS[@]}" \
    model@assim_model="${ASSIM_MODEL}" \
    domain.nx="${NX}" \
    domain.ny="${NY}" \
    domain.nz="${NZ}" \
    ensemble.ensemble_size="${ENSEMBLE_SIZE}" \
    ensemble.num_parallel_processes="${NUM_PARALLEL}" \
    esmda.num_steps="${NUM_ESMDA_STEPS}" \
    obs.interval_seconds="${INTERVAL_SECONDS}" \
    "hydra.run.dir=${RESULTS_DIR}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "Done -- rollout ESMDA outputs under ${RESULTS_DIR}"
