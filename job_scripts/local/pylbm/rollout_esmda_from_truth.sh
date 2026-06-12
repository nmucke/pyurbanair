#!/bin/bash
# LOCAL (no SLURM) rollout-ESMDA-from-truth runner -- pylbm backend ON GPU.
#
# Local sibling of job_scripts/snellius/pylbm/rollout_esmda_from_truth.slurm:
# it runs scripts/run_esmda.py DIRECTLY (no sbatch / module / SLURM env vars),
# keeping all heavy I/O and outputs under the repo (pyurbanair). Run config that
# is shared with the pyudales/pypalm runners lives in ../common.sh (sourced
# below); only the pylbm/GPU specifics are set here.
#
# GPU implies a SINGLE process: the pylbm forward model runs on CUDA
# (assim_model.forward_model.cuda=true) and the ensemble is evaluated with
# ensemble.num_parallel_processes=1 (members run in sequential batches sharing
# the one GPU -- there is no multi-process fan-out as in the CPU/SLURM version).
# Requires a machine with a visible NVIDIA GPU and the `cuda` pixi environment.
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
ENV="${ENV:-cuda}"             # pixi env providing CUDA-enabled deps
ASSIM_MODEL="pylbm"

# pylbm ALWAYS runs on the GPU, and with a single GPU the ensemble is evaluated
# SEQUENTIALLY: one process (num_parallel=1), members in back-to-back batches
# sharing the one device. These are hard-pinned, not env-overridable.
NUM_PARALLEL=1

# Sweep parameters (grid resolution / ensemble / ESMDA steps). Env-overridable so
# the sweep launchers can inject one value per run; each lands in its own RESULTS_DIR.
NX="${NX:-50}"
NY="${NY:-40}"
NZ="${NZ:-16}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-96}"
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-20.0}"   # obs.interval_seconds: time-aggregation bin width [s]

# Shared defaults: paths, domain bounds + sensors, windows, time horizon, dynamic
# parameter settings, localization, ground-truth resolution/validation.
source "${REPO_ROOT}/job_scripts/local/common.sh"

RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}_int${INTERVAL_SECONDS}${LOCALIZATION_TAG}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_TAG}"
RUN_TEMP_DIR="${TEMP_ROOT}/${RUN_TAG}_$$"

mkdir -p "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "LOCAL pylbm GPU rollout-ESMDA on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} (GPU) windows=${NUM_ASSIM_WINDOWS}"
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
  "assim_model.forward_model.cuda=true"
  "paths.experiment_dir=${RUN_TEMP_DIR}"
  "run.ensemble_save_on_disk=true"
)

# COMMON_RUN_FLAGS (from common.sh) carries every shared Hydra override; only the
# assim model, the per-run sweep values, hydra.run.dir and the pylbm/GPU solver
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
