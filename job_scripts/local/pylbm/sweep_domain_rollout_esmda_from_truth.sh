#!/bin/bash
# LOCAL (no SLURM) resolution sweep of rollout-ESMDA-from-truth using the pylbm
# backend ON GPU. This is the local sibling of
# job_scripts/snellius/pylbm/sweep_domain_rollout_esmda_from_truth.sh: instead of
# submitting one SLURM job per resolution, it runs scripts/run_esmda.py DIRECTLY,
# once per resolution, sequentially in this shell.
#
# GPU implies a SINGLE process: the pylbm forward model runs on CUDA
# (assim_model.forward_model.cuda=true) and the ensemble is evaluated with
# ensemble.num_parallel_processes=1 (members run in sequential batches sharing the
# one GPU -- there is no multi-process fan-out as in the CPU/SLURM version).
#
# Run it from anywhere (it cd's to the repo root itself):
#
#     bash job_scripts/local/pylbm/sweep_domain_rollout_esmda_from_truth.sh
#
# Any extra arguments are forwarded as Hydra overrides to EVERY resolution, e.g.:
#
#     bash job_scripts/local/pylbm/sweep_domain_rollout_esmda_from_truth.sh esmda.num_steps=4
#
# Requires a machine with a visible NVIDIA GPU and the `cuda` pixi environment.
set -euo pipefail

# Repo root = two levels up from this script (job_scripts/local/pylbm/ -> repo).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

ASSIM_MODEL="pylbm"

# ============================================================================
# CONFIG -- edit these for your run. (Mirrors the SLURM submit file, minus the
# SLURM/partition/budget machinery.)
# ============================================================================
# Pixi environment that provides CUDA-enabled deps.
ENV="${ENV:-cuda}"

# ROOT directory holding the pre-simulated ground truth. The forward-model script
# nests its output under "<model>_time_varying/", so the actual leaf used is
# ${GROUND_TRUTH_DIR}/${GROUND_TRUTH_MODEL}_time_varying (constructed below). Set
# GROUND_TRUTH_SUBDIR="" if GROUND_TRUTH_DIR already points straight at the leaf.
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-/projects/prjs2075/urbanair/ground_truth_small}"

# Backend that PRODUCED the ground truth (read-only here; sets the obs grid
# mapping). The assimilation backend is fixed to pylbm for this script.
GROUND_TRUTH_MODEL="${GROUND_TRUTH_MODEL:-pyudales}"   # pylbm | pyudales | pypalm

# Where final per-resolution outputs land. Local run -> everything stays under
# the repo (pyurbanair); override to write elsewhere (e.g. a project space).
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/assim_from_ground_truth}"

# Heavy intermediate solver I/O ("scratch"). Local run -> a private, per-run
# folder under the repo instead of the default shared ${cwd}/.temp_palm. This
# replaces the model's .temp via the paths.experiment_dir override below, so
# nothing is written to the repo-root .temp/.temp_palm. Cleaned up on success.
TEMP_ROOT="${TEMP_ROOT:-${REPO_ROOT}/.local_runs/temp}"

# Geometry case bundle (default domain bounds + sensor layout).
CASE="${CASE:-xie_and_castro}"        # xie_and_castro | barcelona

# --- Physical bounds [lo, hi] per axis, in metres: x, y, z. -----------------
X_BOUNDS="[-20.0, 80.0]"
Y_BOUNDS="[0.0, 80.0]"
Z_BOUNDS="[0.0, 32.0]"

# --- Observation sensors (one entry per sensor across the three lists) -------
X_POINTS="[30.0, 60.0, 40.0, 10.0, 65.0]"
Y_POINTS="[10.0, 20.0, 40.0, 60.0, 50.0]"
Z_POINTS="[2.0, 2.0, 2.0, 2.0, 2.0]"

# --- Ensemble / ESMDA -------------------------------------------------------
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-96}"     # number of ensemble members
NUM_PARALLEL=1                           # GPU: single process, sequential batches
NUM_ASSIM_WINDOWS=4                      # number of assimilation windows
NUM_TIME_POINTS=10                       # time-varying param knots per window
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"  # ESMDA iterations per window

# --- Time discretisation ----------------------------------------------------
SIMULATION_TIME=300.0        # per-window length [s]
OUTPUT_FREQUENCY=1.0         # state snapshot interval [s]
SPINUP_TIME=50.0             # constant-inflow plateau before each window [s]

SEED=0
SKIP_VIZ="false"             # set "true" to skip animations/plots

# Whether ESMDA uses correlation localization.
USE_LOCALIZATION="${USE_LOCALIZATION:-false}"
TRUNCATION_CORRELATION=0.3

# ============================================================================
# Resolution sweep. Ground truth is 100 x 80 x 32 (aspect ratio 25:20:8); each
# row keeps that ratio from coarse up to the full truth grid. No wall-clock here
# -- this runs locally, so there is no --time to set.
# Format: "NX NY NZ"   (each row runs one separate, sequential local run)
# ============================================================================
RESOLUTIONS=(
  "25 20 8"     # k=1  (coarsest)
  "50 40 16"    # k=2
  "75 60 24"    # k=3
  "100 80 32"   # k=4  (== ground-truth resolution)
)

# --- Localization flags + output tag ----------------------------------------
case "${USE_LOCALIZATION}" in
  true|True|TRUE|1|yes)
    LOCALIZATION_TAG="_localization"
    LOCALIZATION_FLAGS=(
      "++esmda.localization._target_=data_assimilation.localization.correlation.CorrelationLocalization"
      "++esmda.localization.truncation_correlation=${TRUNCATION_CORRELATION}"
      "++esmda.localization.tapering_beta=0.5"
      "++esmda.localization.max_inflation=8.0"
      "++esmda.localization.block_grouping=true"
    )
    ;;
  false|False|FALSE|0|no)
    LOCALIZATION_TAG=""
    LOCALIZATION_FLAGS=( "esmda.localization=null" )
    ;;
  *)
    echo "error: USE_LOCALIZATION must be true/false (got '${USE_LOCALIZATION}')" >&2
    exit 1
    ;;
esac

# --- Resolve + validate the ground-truth leaf -------------------------------
GROUND_TRUTH_SUBDIR="${GROUND_TRUTH_SUBDIR-${GROUND_TRUTH_MODEL}_time_varying}"
if [ -n "${GROUND_TRUTH_SUBDIR}" ]; then
  GROUND_TRUTH_PATH="${GROUND_TRUTH_DIR}/${GROUND_TRUTH_SUBDIR}"
else
  GROUND_TRUTH_PATH="${GROUND_TRUTH_DIR}"
fi
[ -f "${GROUND_TRUTH_PATH}/state.nc" ] && [ -f "${GROUND_TRUTH_PATH}/params.nc" ] || {
  echo "error: expected state.nc + params.nc in ${GROUND_TRUTH_PATH}" >&2
  echo "       (set GROUND_TRUTH_DIR / GROUND_TRUTH_SUBDIR to the right location)" >&2
  exit 1
}
case "${GROUND_TRUTH_MODEL}" in
  pylbm|pyudales|pypalm) ;;
  *) echo "unknown GROUND_TRUTH_MODEL '${GROUND_TRUTH_MODEL}'" >&2; exit 1 ;;
esac

# Keep BLAS single-threaded; line-buffer Python output.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "LOCAL pylbm GPU rollout-ESMDA sweep -- ${#RESOLUTIONS[@]} resolutions, run sequentially."
echo "Repo root:    ${REPO_ROOT}"
echo "Pixi env:     ${ENV}"
echo "Ground truth: ${GROUND_TRUTH_PATH} (truth model=${GROUND_TRUTH_MODEL})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} (GPU) windows=${NUM_ASSIM_WINDOWS} steps=${NUM_ESMDA_STEPS}"
[ "$#" -gt 0 ] && echo "Forwarding extra Hydra overrides to every run: $*"

for res in "${RESOLUTIONS[@]}"; do
  read -r nx ny nz <<<"${res}"

  RUN_TAG="${ASSIM_MODEL}_nx${nx}_ny${ny}_nz${nz}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}${LOCALIZATION_TAG}"
  RESULTS_DIR="${RESULTS_ROOT}/${RUN_TAG}"
  # No SLURM job id locally: make the scratch unique per resolution (RUN_TAG) and
  # per invocation (this shell's PID, $$) so concurrent runs never collide.
  RUN_TEMP_DIR="${TEMP_ROOT}/${RUN_TAG}_$$"
  # Remove this run's scratch on exit (success or failure); harmless if missing.
  trap 'rm -rf "${RUN_TEMP_DIR}"' EXIT

  echo
  echo "==> NX=${nx} NY=${ny} NZ=${nz}  ->  ${RESULTS_DIR}"
  mkdir -p "${RESULTS_DIR}" "${RUN_TEMP_DIR}"

  # The LBM Fortran build mutates its own source tree (mod_dimensions.F90,
  # generated m_*.F90, the makefile) and writes objects + the boltzmann binary in
  # place. That tree is the shared in-repo submodule, so give this run a private
  # copy on scratch and point pylbm at it via PYLBM_LBM_PATH.
  JOB_LBM_DIR="${RUN_TEMP_DIR}/LBM"
  rsync -a --delete --exclude='.git' libs/pylbm/LBM/ "${JOB_LBM_DIR}/"
  export PYLBM_LBM_PATH="${JOB_LBM_DIR}"

  pixi run -e "${ENV}" -- python -u \
      scripts/run_esmda.py \
      esmda/smoother=dynamic \
      model@truth_model="${GROUND_TRUTH_MODEL}" \
      model@assim_model="${ASSIM_MODEL}" \
      params@truth_params=dynamic_truth \
      params@prior_params=dynamic \
      case="${CASE}" \
      domain.nx="${nx}" \
      domain.ny="${ny}" \
      domain.nz="${nz}" \
      "domain.bounds=[${X_BOUNDS}, ${Y_BOUNDS}, ${Z_BOUNDS}]" \
      obs.mode=points \
      "obs.x_points=${X_POINTS}" \
      "obs.y_points=${Y_POINTS}" \
      "obs.z_points=${Z_POINTS}" \
      time.simulation_time="${SIMULATION_TIME}" \
      time.output_frequency="${OUTPUT_FREQUENCY}" \
      time.spinup_time="${SPINUP_TIME}" \
      prior_params.time_coords.num="${NUM_TIME_POINTS}" \
      params.time_coords.num="${NUM_TIME_POINTS}" \
      ensemble.ensemble_size="${ENSEMBLE_SIZE}" \
      ensemble.num_parallel_processes="${NUM_PARALLEL}" \
      esmda.num_assimilation_windows="${NUM_ASSIM_WINDOWS}" \
      esmda.num_steps="${NUM_ESMDA_STEPS}" \
      "${LOCALIZATION_FLAGS[@]}" \
      esmda.seed="${SEED}" \
      run.truth_dir="${GROUND_TRUTH_PATH}" \
      run.skip_viz="${SKIP_VIZ}" \
      "hydra.run.dir=${RESULTS_DIR}" \
      assim_model.forward_model.cuda=true \
      "paths.experiment_dir=${RUN_TEMP_DIR}" \
      "$@"

  rm -rf "${RUN_TEMP_DIR}"
  echo "==> done NX=${nx} NY=${ny} NZ=${nz} -- outputs under ${RESULTS_DIR}"
done

echo
echo "All ${#RESOLUTIONS[@]} resolutions complete. Outputs under ${RESULTS_ROOT}"
