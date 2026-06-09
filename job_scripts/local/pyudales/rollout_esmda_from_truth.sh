#!/bin/bash
# LOCAL (no SLURM) rollout-ESMDA-from-truth runner -- pyudales backend.
#
# Local sibling of job_scripts/snellius/pyudales/rollout_esmda_from_truth.slurm:
# it runs scripts/run_esmda.py DIRECTLY (no sbatch / module / SLURM env vars),
# keeping all heavy I/O and outputs under the repo (pyurbanair). The run config
# below is identical to the Snellius submit file -- only the execution machinery
# (SLURM, /scratch-shared, /projects storage) is swapped for local equivalents.
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

# ============================================================================
# CONFIG -- edit these for your run. (Mirrors the SLURM submit file; the
# SLURM/partition/budget machinery is gone.)
# ============================================================================
# Pixi environment. pyudales is CPU-only; "dev" carries every solver feature.
ENV="${ENV:-dev}"

# ROOT directory holding the pre-simulated ground truth (leaf is
# ${GROUND_TRUTH_DIR}/${GROUND_TRUTH_MODEL}_time_varying; set GROUND_TRUTH_SUBDIR=""
# if GROUND_TRUTH_DIR already points straight at the leaf).
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-/projects/prjs2075/urbanair/ground_truth_small}"
GROUND_TRUTH_MODEL="${GROUND_TRUTH_MODEL:-pyudales}"   # pylbm | pyudales | pypalm
ASSIM_MODEL="pyudales"

# Local run -> outputs and scratch stay under the repo (pyurbanair). No SLURM job
# id locally, so scratch is made unique per run from RUN_TAG + this shell's PID.
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/assim_from_ground_truth}"
TEMP_ROOT="${TEMP_ROOT:-${REPO_ROOT}/.local_runs/temp}"

# Geometry case bundle (default domain bounds + sensor layout).
CASE="${CASE:-xie_and_castro}"        # xie_and_castro | barcelona

# --- Domain (grid resolution + physical bounds) -----------------------------
NX="${NX:-50}"
NY="${NY:-40}"
NZ="${NZ:-16}"
X_BOUNDS="[-20.0, 80.0]"
Y_BOUNDS="[0.0, 80.0]"
Z_BOUNDS="[0.0, 32.0]"

# --- Observation sensors (one entry per sensor across the three lists) -------
X_POINTS="[30.0, 60.0, 40.0, 10.0, 65.0]"
Y_POINTS="[10.0, 20.0, 40.0, 60.0, 50.0]"
Z_POINTS="[2.0, 2.0, 2.0, 2.0, 2.0]"

# --- Ensemble / ESMDA -------------------------------------------------------
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-96}"     # number of ensemble members
NUM_PARALLEL="${NUM_PARALLEL:-}"         # empty -> min(ENSEMBLE_SIZE, local cores)
NUM_ASSIM_WINDOWS=4
NUM_TIME_POINTS=10
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"

# --- Time discretisation ----------------------------------------------------
SIMULATION_TIME=300.0
OUTPUT_FREQUENCY=1.0
SPINUP_TIME=50.0

SEED=0
SKIP_VIZ="false"

USE_LOCALIZATION="${USE_LOCALIZATION:-false}"
TRUNCATION_CORRELATION=0.3
# ============================================================================

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

RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}${LOCALIZATION_TAG}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_TAG}"
RUN_TEMP_DIR="${TEMP_ROOT}/${RUN_TAG}_$$"

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
for m in "${GROUND_TRUTH_MODEL}" "${ASSIM_MODEL}"; do
  case "${m}" in
    pylbm|pyudales|pypalm) ;;
    *) echo "unknown model '${m}' (expected pylbm|pyudales|pypalm)" >&2; exit 1 ;;
  esac
done

# Worker count: default to one per ensemble member, capped at the local cores.
CORES="${LOCAL_CORES:-$(nproc)}"
if [ -z "${NUM_PARALLEL}" ]; then
  NUM_PARALLEL=$(( ENSEMBLE_SIZE < CORES ? ENSEMBLE_SIZE : CORES ))
fi
if (( NUM_PARALLEL > CORES )); then
  echo "warning: NUM_PARALLEL=${NUM_PARALLEL} exceeds local cores=${CORES}; capping." >&2
  NUM_PARALLEL=${CORES}
fi

mkdir -p "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "LOCAL pyudales rollout-ESMDA on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} windows=${NUM_ASSIM_WINDOWS} cores=${CORES}"
echo "ESMDA steps=${NUM_ESMDA_STEPS} localization=${USE_LOCALIZATION}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

# Keep BLAS single-threaded; line-buffer Python output.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# pyudales scratch lands under this run's private temp dir.
EXTRA_FLAGS=(
  "assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}"
  "+assim_model.forward_model.output_dir=${RUN_TEMP_DIR}/outputs"
)

pixi run -e "${ENV}" -- python -u \
    scripts/run_esmda.py \
    esmda/smoother=dynamic \
    model@truth_model="${GROUND_TRUTH_MODEL}" \
    model@assim_model="${ASSIM_MODEL}" \
    params@truth_params=dynamic_truth \
    params@prior_params=dynamic \
    case="${CASE}" \
    domain.nx="${NX}" \
    domain.ny="${NY}" \
    domain.nz="${NZ}" \
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
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "Done -- rollout ESMDA outputs under ${RESULTS_DIR}"
