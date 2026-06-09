#!/bin/bash
# Run time-varying-parameter ROLLOUT ESMDA against a pre-simulated ground truth
# on Snellius, assimilating with the pylbm backend.
#
# This is a SELF-CONTAINED submit file: edit the CONFIG block below, then submit
# it directly:
#
#     sbatch job_scripts/snellius/pylbm/rollout_esmda_from_truth.slurm
#
# Extra Hydra overrides may also be appended and take precedence, e.g.:
#
#     sbatch job_scripts/snellius/pylbm/rollout_esmda_from_truth.slurm esmda.num_steps=4
#
# It drives scripts/run_esmda.py in its loaded-truth mode (selected by
# run.truth_dir): instead of simulating the truth it LOADS the ground truth
# (state.nc + params.nc) from GROUND_TRUTH_DIR, chops it into assimilation
# windows, and runs the time-varying (dynamic) smoother
# (esmda/smoother=dynamic -> TimeVaryingParameterESMDA). The number of windows
# is fixed by esmda.num_assimilation_windows; keep simulation_time * windows <=
# the time length of the loaded ground truth.
#
# The TRUTH model (model@truth_model) is only read here -- its forward model is
# never run -- so only its solver_name matters (it must match the backend that
# PRODUCED the ground truth, so the observation grid mapping is correct). The
# ASSIM model (model@assim_model, fixed to pylbm in this folder) is the
# one actually integrated across the ensemble.
#
# All three backend folders (pyudales/ pypalm/ pylbm/) carry the same sweep
# launchers and point at the SAME ground truth -- only ASSIM_MODEL differs -- so
# the runs are directly comparable.
#
# NB: NUM_PARALLEL auto-tracks ENSEMBLE_SIZE but is capped at --cpus-per-task
# (one core per worker); raise --cpus-per-task to run a larger ensemble fully in
# parallel.
#SBATCH --job-name=rollout_pylbm
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=1750M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-rollout_pylbm-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-rollout_pylbm-%j.err

set -euo pipefail

# ============================================================================
# CONFIG -- edit these for your run.
# ============================================================================
# ROOT directory holding the pre-simulated ground truth. The forward-model
# script nests its output under "<model>_time_varying/", so the actual leaf used
# is ${GROUND_TRUTH_DIR}/${GROUND_TRUTH_MODEL}_time_varying (constructed below).
# If you instead point this straight at the leaf (the folder that directly
# contains state.nc + params.nc), set GROUND_TRUTH_SUBDIR="" to disable nesting.
GROUND_TRUTH_DIR="/projects/prjs2075/urbanair/ground_truth_small"

# Backend that PRODUCED the ground truth (read-only here; sets the obs grid
# mapping). The SAME truth is shared by the pyudales/ pypalm/ pylbm/ folders.
GROUND_TRUTH_MODEL="pyudales"    # pylbm | pyudales | pypalm
# Backend used to ASSIMILATE -- fixed per folder.
ASSIM_MODEL="pylbm"

# Geometry case bundle (default domain bounds + sensor layout).
CASE="xie_and_castro"         # xie_and_castro | barcelona

# --- Domain arguments (grid resolution + physical bounds) -------------------
# NX/NY/NZ default to the values below but may be overridden from the
# environment (e.g. by the sweep launchers, which submit one job per resolution).
# Because RUN_TAG/RESULTS_DIR below embed them, each resolution lands in its own
# output subdir automatically.
NX="${NX:-50}"
NY="${NY:-40}"
NZ="${NZ:-16}"
# Physical bounds as [lo, hi] per axis, in metres: x, y, z.
X_BOUNDS="[-20.0, 80.0]"
Y_BOUNDS="[0.0, 80.0]"
Z_BOUNDS="[0.0, 32.0]"

# --- Observation sensors ----------------------------------------------------
# Sensor coordinates [m], one entry per sensor across the three lists. They MUST
# lie inside the domain AND inside the solver's grid. For uDALES the velocity
# components live on STAGGERED faces whose last node sits one cell inside the
# domain max, so keep sensors strictly interior.
# X_POINTS="[20.0, 20.0, 38.0]"
# Y_POINTS="[20.0, 10.0, 10.0]"
# Z_POINTS="[3.0, 3.0, 3.0]"

X_POINTS="[30.0, 60.0, 40.0, 10.0, 65.0]"
Y_POINTS="[10.0, 20.0, 40.0, 60.0, 50.0]"
Z_POINTS="[2.0, 2.0, 2.0, 2.0, 2.0]"

# --- Ensemble / ESMDA -------------------------------------------------------
# Overridable from the environment (the sweep launchers set it together with
# --cpus-per-task so there is exactly one core per ensemble member).
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-96}"             # number of ensemble members
# Parallel workers. Leave empty to auto-track ENSEMBLE_SIZE (capped at the
# allocated --cpus-per-task); set a number to pin it explicitly.
# NUM_PARALLEL=""
# NUM_ASSIM_WINDOWS=5          # number of assimilation windows
# NUM_TIME_POINTS=6            # time-varying parameter knots per window (prior)
NUM_PARALLEL=""
NUM_ASSIM_WINDOWS=4          # number of assimilation windows
NUM_TIME_POINTS=10            # time-varying param knots per window (sets the prior AND
# Overridable from the environment (the ESMDA-steps sweep launcher sets it); it
# is folded into RUN_TAG below so each value lands in its own output subdir.
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"            # ESMDA iterations per window

# --- Time discretisation ----------------------------------------------------
# Per-window length / snapshot interval for the ASSIM model. These need NOT match
# the ground-truth generation: run_esmda.py aggregates observations into
# obs.interval_seconds bins, so truth and assim frames align regardless of their
# raw output cadence. Keep SIMULATION_TIME * NUM_ASSIM_WINDOWS <= the truth
# length.
# SIMULATION_TIME=180.0        # per-window length [s]
# OUTPUT_FREQUENCY=5.0         # state snapshot interval [s]
# SPINUP_TIME=10.0             # constant-inflow plateau before each window [s]

SIMULATION_TIME=300.0        # per-window length [s]
OUTPUT_FREQUENCY=1.0         # state snapshot interval [s]
SPINUP_TIME=50.0             # constant-inflow plateau before each window [s]

SEED=0
SKIP_VIZ="false"             # set "true" to skip animations/plots

# Whether ESMDA uses correlation localization. Env-overridable.
#   true  -> localized update with TRUNCATION_CORRELATION (tag "_localization")
#   false -> global (unlocalized) Kalman update (the config default)
USE_LOCALIZATION="${USE_LOCALIZATION:-false}"
# Correlation-localization truncation; only used when USE_LOCALIZATION=true.
TRUNCATION_CORRELATION=0.3
# ============================================================================

ENV="snellius"
PROJECT_STORAGE="/projects/prjs2075/urbanair"

# Localization: build the matching Hydra override and an output-dir tag. The
# config defaults esmda.localization=null (global update). When enabled we add
# the full CorrelationLocalization block via "++" (the commented-out block in
# conf/esmda.yaml is not composed, so we set every field explicitly) and tag
# "_localization"; when disabled we pin esmda.localization=null and add no tag.
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

# Final outputs land under "assim_from_ground_truth", in a subdir tagged with the
# assim model and the run's domain / ensemble / step settings so different
# configurations (and the three backend folders) don't overwrite each other.
RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}${LOCALIZATION_TAG}"
RESULTS_DIR="${PROJECT_STORAGE}/assim_from_ground_truth/${RUN_TAG}"

# Select the ground-truth leaf that matches the chosen truth backend. Override
# with GROUND_TRUTH_SUBDIR (e.g. "" if GROUND_TRUTH_DIR is already the leaf).
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

# Worker count: default to one per ensemble member, capped at the cores the
# allocation actually gave us. A larger ENSEMBLE_SIZE than cores still runs --
# the extra members are processed in batches.
CORES="${SLURM_CPUS_PER_TASK:-1}"
if [ -z "${NUM_PARALLEL}" ]; then
  NUM_PARALLEL=$(( ENSEMBLE_SIZE < CORES ? ENSEMBLE_SIZE : CORES ))
fi
if (( NUM_PARALLEL > CORES )); then
  echo "warning: NUM_PARALLEL=${NUM_PARALLEL} exceeds --cpus-per-task=${CORES}; capping to ${CORES}." >&2
  NUM_PARALLEL=${CORES}
fi

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/snellius/out_files "${RESULTS_DIR}"

# Heavy intermediate solver I/O on /scratch-shared, cleaned up on success.
RUN_TEMP_DIR="/scratch-shared/${USER}/urbanair_temp/${SLURM_JOB_ID}"
mkdir -p "${RUN_TEMP_DIR}"
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "Job ${SLURM_JOB_ID} on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Bounds: x=${X_BOUNDS} y=${Y_BOUNDS} z=${Z_BOUNDS}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} windows=${NUM_ASSIM_WINDOWS} cores=${SLURM_CPUS_PER_TASK}"
echo "ESMDA steps=${NUM_ESMDA_STEPS} localization=${USE_LOCALIZATION} (${LOCALIZATION_FLAGS[*]})"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# One core per ensemble worker; keep BLAS single-threaded so workers don't fight
# for the allocated cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Per-model solver flags for the ASSIM model only -- the truth model is read from
# disk, never integrated, so its forward-model config is left untouched.
EXTRA_FLAGS=()
case "${ASSIM_MODEL}" in
  pylbm)
    EXTRA_FLAGS+=( "assim_model.forward_model.cuda=false" )
    # pylbm's temp_dir defaults to ${paths.experiment_dir} (conf/paths.yaml),
    # which is SHARED across every job launched from this repo. Override that
    # config key so this job's experiment/ensemble I/O lands in its own scratch.
    EXTRA_FLAGS+=( "paths.experiment_dir=${RUN_TEMP_DIR}" )
    # The LBM Fortran build mutates its own source tree (mod_dimensions.F90,
    # generated m_*.F90, the makefile) and writes objects + the boltzmann binary
    # in place. That tree is the shared in-repo submodule, so concurrent `make`
    # runs corrupt each other. Give this job a private copy on scratch and point
    # pylbm at it via PYLBM_LBM_PATH (read in libs/pylbm/.../__init__.py).
    JOB_LBM_DIR="${RUN_TEMP_DIR}/LBM"
    rsync -a --delete --exclude='.git' libs/pylbm/LBM/ "${JOB_LBM_DIR}/"
    export PYLBM_LBM_PATH="${JOB_LBM_DIR}"
    ;;
  pyudales)
    EXTRA_FLAGS+=( "assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
    EXTRA_FLAGS+=( "+assim_model.forward_model.output_dir=${RUN_TEMP_DIR}/outputs" )
    ;;
  pypalm)
    EXTRA_FLAGS+=( "assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
    export PYURBANAIR_DISABLE_CPU_PINNING=1
    export OMPI_MCA_mpi_yield_when_idle=1
    export OMPI_MCA_rmaps_base_oversubscribe=true
    export OMPI_MCA_hwloc_base_binding_policy=none
    export PYPALM_USE_DIRECT_RUN="${PYPALM_USE_DIRECT_RUN:-1}"
    export PYPALM_FAST_IO_CATALOG="${TMPDIR}/urbanair_palm_${SLURM_JOB_ID}"
    mkdir -p "${PYPALM_FAST_IO_CATALOG}"
    ;;
esac

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
