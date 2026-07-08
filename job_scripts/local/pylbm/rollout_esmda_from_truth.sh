#!/bin/bash
# LOCAL (no SLURM) rollout-ESMDA-from-truth runner -- pylbm backend (GPU or CPU).
#
# Local sibling of job_scripts/snellius/pylbm/rollout_esmda_from_truth.slurm:
# it runs scripts/esmda/run_esmda.py DIRECTLY (no sbatch / module / SLURM env vars),
# keeping all heavy I/O and outputs under the repo (pyurbanair). Run config that
# is shared with the pyudales/pypalm runners lives in ../common.sh (sourced
# below); only the pylbm specifics are set here.
#
# Device is chosen with USE_CUDA (default true):
#   USE_CUDA=true  -- CUDA forward model (assim_model.forward_model.cuda=true) in
#                     the `cuda` pixi env. One GPU => a SINGLE process: the
#                     ensemble is evaluated with num_parallel_processes=1 (members
#                     in sequential batches sharing the device). Needs a visible
#                     NVIDIA GPU.
#   USE_CUDA=false -- CPU forward model in the `dev` pixi env, with the ensemble
#                     fanned out across up to LOCAL_MAX_PARALLEL worker processes
#                     (cap with NUM_PARALLEL=…), exactly like the pyudales runner.
#                     Use this on a laptop with no NVIDIA GPU, e.g.:
#                         USE_CUDA=false LOCAL_MAX_PARALLEL=8 bash …
#
# Run it from anywhere (it cd's to the repo root itself):
#
#     bash job_scripts/local/pylbm/rollout_esmda_from_truth.sh
#
# Extra Hydra overrides may be appended and take precedence, e.g.:
#
#     bash job_scripts/local/pylbm/rollout_esmda_from_truth.sh esmda.num_steps=4
#
# NX/NY/NZ, ENSEMBLE_SIZE and NUM_ESMDA_STEPS are read from the environment (the
# sweep launchers in this folder set them), so each configuration lands in its
# own RESULTS_DIR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Backend-specific knobs ------------------------------------------------------
ASSIM_MODEL="pylbm"

# Device toggle. GPU (default): CUDA forward model, `cuda` pixi env. CPU
# (USE_CUDA=false): CPU forward model, `dev` pixi env. ENV may still be overridden
# explicitly; otherwise it defaults to the env matching the chosen device.
case "${USE_CUDA:-true}" in
  true|True|TRUE|1|yes)   USE_CUDA=true ; DEVICE_LABEL="GPU" ; ENV="${ENV:-cuda}" ;;
  false|False|FALSE|0|no) USE_CUDA=false; DEVICE_LABEL="CPU" ; ENV="${ENV:-dev}"  ;;
  *) echo "error: USE_CUDA must be true/false (got '${USE_CUDA}')" >&2; exit 1 ;;
esac

# Worker count. GPU is hard-pinned to a SINGLE process (one device, ensemble
# evaluated in sequential batches). CPU fans the ensemble out across up to
# LOCAL_MAX_PARALLEL worker processes (your chosen ceiling, default 16; set
# LOCAL_MAX_PARALLEL=… or pin an exact count with NUM_PARALLEL=…), capped at the
# ensemble size. Resolved below, once common.sh has provided LOCAL_MAX_PARALLEL.
NUM_PARALLEL="${NUM_PARALLEL:-}"

# Sweep parameters (grid resolution / ensemble / ESMDA steps). Env-overridable so
# the sweep launchers can inject one value per run; each lands in its own RESULTS_DIR.
NX="${NX:-50}"
NY="${NY:-40}"
NZ="${NZ:-16}"
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"
# ENSEMBLE_SIZE and INTERVAL_SECONDS default in common.sh (still env-overridable).

# Shared defaults: paths, domain bounds + sensors, windows, time horizon, dynamic
# parameter settings, localization, ground-truth resolution/validation.
source "${REPO_ROOT}/job_scripts/local/common.sh"

# Resolve worker count now that LOCAL_MAX_PARALLEL is in scope (see note above).
if [ "${USE_CUDA}" = true ]; then
  NUM_PARALLEL=1
elif [ -z "${NUM_PARALLEL}" ]; then
  NUM_PARALLEL=$(( ENSEMBLE_SIZE < LOCAL_MAX_PARALLEL ? ENSEMBLE_SIZE : LOCAL_MAX_PARALLEL ))
fi

RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}_int${INTERVAL_SECONDS}${LOCALIZATION_TAG}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_TAG}"
RUN_TEMP_DIR="${TEMP_ROOT}/${RUN_TAG}_$$"

mkdir -p "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "LOCAL pylbm ${DEVICE_LABEL} rollout-ESMDA on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} (${DEVICE_LABEL}) windows=${NUM_ASSIM_WINDOWS}"
echo "ESMDA steps=${NUM_ESMDA_STEPS} obs_interval=${INTERVAL_SECONDS}s localization=${USE_LOCALIZATION}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

# The LBM Fortran build mutates its own source tree (mod_dimensions.F90,
# generated m_*.F90, the makefile) and writes objects + the boltzmann binary in
# place. That tree is the shared in-repo submodule, so give this run a private
# copy on scratch and point pylbm at it via PYLBM_LBM_PATH.
JOB_LBM_DIR="${RUN_TEMP_DIR}/LBM"
rsync -a --delete --exclude='.git' libs/pylbm/LBM/ "${JOB_LBM_DIR}/"
export PYLBM_LBM_PATH="${JOB_LBM_DIR}"

# pylbm runs on GPU here; its experiment/ensemble scratch lands under this run's
# private temp dir (overrides paths.experiment_dir, which otherwise defaults to a
# shared in-repo path).
#
# ensemble_save_on_disk=true: the ensemble forecasts are written one NetCDF per
# member (per ESMDA step) instead of being concatenated into a single in-memory
# ensemble Dataset. A single ensemble state is ~29 GB at 75x60x24 and ~70 GB at
# 100x80x32, so the in-memory path (which also keeps num_steps+1 of them for the
# state history) overruns host RAM and the run is OOM-killed mid-window. On the
# GPU the ensemble is already evaluated sequentially (num_parallel=1), so the
# disk path applies cleanly; run_esmda.py reassembles the per-window
# prior/posterior states by streaming the per-member files.
EXTRA_FLAGS=(
  "assim_model.forward_model.cuda=${USE_CUDA}"
  "paths.experiment_dir=${RUN_TEMP_DIR}"
  "run.ensemble_save_on_disk=true"
)

# COMMON_RUN_FLAGS (from common.sh) carries every shared Hydra override; only the
# assim model, the per-run sweep values, hydra.run.dir and the pylbm/GPU solver
# flags are added here.
pixi run -e "${ENV}" -- python -u \
    scripts/esmda/run_esmda.py \
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
