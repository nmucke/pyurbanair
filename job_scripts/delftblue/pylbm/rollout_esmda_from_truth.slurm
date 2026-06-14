#!/bin/bash
# DELFTBLUE rollout-ESMDA-from-truth runner -- pylbm backend (CPU).
#
# DelftBlue sibling of job_scripts/local/pylbm/rollout_esmda_from_truth.sh, BUT on
# the DelftBlue CPU compute partition pylbm runs on the CPU (cuda=false), fanning
# the ensemble out across parallel processes -- it does NOT use the GPU/single-
# process path of the local runner. It submits scripts/run_esmda.py in its
# loaded-truth mode (run.truth_dir). Run config shared with the pyudales/pypalm
# runners lives in ../common.sh (sourced below); only the pylbm specifics here.
#
# Submit it directly (edit ../common.sh to retune the shared config), or append
# Hydra overrides which take precedence:
#
#     sbatch job_scripts/delftblue/pylbm/rollout_esmda_from_truth.slurm
#     sbatch job_scripts/delftblue/pylbm/rollout_esmda_from_truth.slurm esmda.num_steps=4
#
# NX/NY/NZ, ENSEMBLE_SIZE and NUM_ESMDA_STEPS are read from the environment (the
# sweep launchers in this folder inject one value per job via --export), so each
# configuration lands in its own RESULTS_DIR.
#
# No parallel cap: num_parallel == ensemble size; the sweep launchers request one
# core per ensemble member (--cpus-per-task == ENSEMBLE_SIZE, capped at a 64-core
# node -- larger ensembles oversubscribe).
#SBATCH --job-name=rollout_pylbm
#SBATCH --partition=compute-p2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3G
#SBATCH --time=16:00:00
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-rollout_pylbm-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-rollout_pylbm-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Backend-specific knobs ------------------------------------------------------
ENV="delftblue"
ASSIM_MODEL="pylbm"

# Sweep parameters (grid resolution / ensemble / ESMDA steps). Env-overridable so
# the sweep launchers can inject one value per job; each lands in its own RESULTS_DIR.
NX="${NX:-50}"
NY="${NY:-40}"
NZ="${NZ:-16}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-96}"
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-20.0}"   # obs.interval_seconds: time-aggregation bin width [s]

# Shared defaults: paths, domain bounds + sensors, windows, time horizon, dynamic
# parameter settings, localization, ground-truth resolution/validation, MPI env,
# and the COMMON_RUN_FLAGS array of every shared Hydra override.
source "${REPO_ROOT}/job_scripts/delftblue/common.sh"

# No parallel cap: one worker per ensemble member (the sweep launchers request
# --cpus-per-task == ENSEMBLE_SIZE up to a node). Pin an exact count with NUM_PARALLEL=….
NUM_PARALLEL="${NUM_PARALLEL:-${ENSEMBLE_SIZE}}"

RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}_int${INTERVAL_SECONDS}${LOCALIZATION_TAG}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_TAG}"
RUN_TEMP_DIR="${TEMP_ROOT}/${SLURM_JOB_ID:-$$}"

mkdir -p job_scripts/delftblue/out_files "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "DELFTBLUE pylbm rollout-ESMDA: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} windows=${NUM_ASSIM_WINDOWS} cores=${SLURM_CPUS_PER_TASK:-?}"
echo "ESMDA steps=${NUM_ESMDA_STEPS} obs_interval=${INTERVAL_SECONDS}s localization=${USE_LOCALIZATION}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# The LBM Fortran build mutates its own source tree (mod_dimensions.F90, generated
# m_*.F90, the makefile) and writes objects + the boltzmann binary in place. That
# tree is the shared in-repo submodule, so give this job a private copy on scratch
# and point pylbm at it via PYLBM_LBM_PATH (read in libs/pylbm/.../__init__.py).
JOB_LBM_DIR="${RUN_TEMP_DIR}/LBM"
rsync -a --delete --exclude='.git' libs/pylbm/LBM/ "${JOB_LBM_DIR}/"
export PYLBM_LBM_PATH="${JOB_LBM_DIR}"

# pylbm on the CPU compute partition (cuda=false). Its experiment/ensemble scratch
# (paths.experiment_dir, otherwise a SHARED in-repo path) lands under this job's
# private temp dir.
EXTRA_FLAGS=(
  "assim_model.forward_model.cuda=false"
  "paths.experiment_dir=${RUN_TEMP_DIR}"
)

# COMMON_RUN_FLAGS (from common.sh) carries every shared Hydra override; only the
# assim model, the per-run sweep values, hydra.run.dir and the pylbm solver flags
# are added here.
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
