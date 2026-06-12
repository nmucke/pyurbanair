#!/bin/bash
# SNELLIUS rollout-ESMDA-from-truth runner -- pyudales backend (CPU).
#
# Snellius sibling of job_scripts/local/pyudales/rollout_esmda_from_truth.sh:
# it submits scripts/run_esmda.py in its loaded-truth mode (run.truth_dir),
# loading a pre-simulated ground truth and running the time-varying (dynamic)
# smoother over esmda.num_assimilation_windows windows. Run config that is shared
# with the pylbm/pypalm runners lives in ../common.sh (sourced below); only the
# pyudales/CPU specifics are set here.
#
# Submit it directly (edit ../common.sh to retune the shared config), or append
# Hydra overrides which take precedence:
#
#     sbatch job_scripts/snellius/pyudales/rollout_esmda_from_truth.slurm
#     sbatch job_scripts/snellius/pyudales/rollout_esmda_from_truth.slurm esmda.num_steps=4
#
# NX/NY/NZ, ENSEMBLE_SIZE and NUM_ESMDA_STEPS are read from the environment (the
# sweep launchers in this folder inject one value per job via --export), so each
# configuration lands in its own RESULTS_DIR.
#
# No parallel cap on Snellius: num_parallel == ensemble size, and the sweep
# launchers request one core per ensemble member (--cpus-per-task == ENSEMBLE_SIZE).
#SBATCH --job-name=rollout_pyudales
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=1750M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-rollout_pyudales-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-rollout_pyudales-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Backend-specific knobs ------------------------------------------------------
ENV="snellius"
ASSIM_MODEL="pyudales"

# Sweep parameters (grid resolution / ensemble / ESMDA steps). Env-overridable so
# the sweep launchers can inject one value per job; each lands in its own RESULTS_DIR.
NX="${NX:-50}"
NY="${NY:-40}"
NZ="${NZ:-16}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-96}"
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-20.0}"   # obs.interval_seconds: time-aggregation bin width [s]

# Shared defaults: paths, domain bounds + sensors, windows, time horizon, dynamic
# parameter settings, localization, ground-truth resolution/validation, and the
# COMMON_RUN_FLAGS array of every shared Hydra override.
source "${REPO_ROOT}/job_scripts/snellius/common.sh"

# No parallel cap: one worker per ensemble member (the sweep launchers request
# --cpus-per-task == ENSEMBLE_SIZE). Pin an exact count with NUM_PARALLEL=….
NUM_PARALLEL="${NUM_PARALLEL:-${ENSEMBLE_SIZE}}"

RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}_int${INTERVAL_SECONDS}${LOCALIZATION_TAG}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_TAG}"
RUN_TEMP_DIR="${TEMP_ROOT}/${SLURM_JOB_ID:-$$}"

mkdir -p job_scripts/snellius/out_files "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "SNELLIUS pyudales rollout-ESMDA: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} windows=${NUM_ASSIM_WINDOWS} cores=${SLURM_CPUS_PER_TASK:-?}"
echo "ESMDA steps=${NUM_ESMDA_STEPS} obs_interval=${INTERVAL_SECONDS}s localization=${USE_LOCALIZATION}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# pyudales scratch lands under this job's private temp dir.
# ensemble_save_on_disk=true (matches the local runner): write the ensemble
# forecasts one NetCDF per member (per ESMDA step) instead of concatenating the
# full ensemble field into one in-memory Dataset. A single ensemble state is
# ~29 GB at 75x60x24 and ~70 GB at 100x80x32 (ens=96), so the in-memory path
# overruns the allocation's RAM (1750M/core) at the sweep's high-resolution
# rows. run_esmda.py reassembles the per-window prior/posterior states by
# streaming the per-member files; diverged members are resampled on disk (the
# parallel path included).
EXTRA_FLAGS=(
  "assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}"
  "+assim_model.forward_model.output_dir=${RUN_TEMP_DIR}/outputs"
  "run.ensemble_save_on_disk=true"
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
