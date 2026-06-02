#!/bin/bash
# Run time-varying-parameter ROLLOUT ESMDA against a pre-simulated ground truth
# on Snellius. Like job_scripts/snellius/ground_truth.slurm, this is a
# SELF-CONTAINED submit file: edit the CONFIG block below, then submit directly:
#
#     sbatch job_scripts/snellius/rollout_esmda_from_truth.slurm
#
# Extra Hydra overrides may also be appended and take precedence, e.g.:
#
#     sbatch job_scripts/snellius/rollout_esmda_from_truth.slurm esmda.num_steps=4
#
# It drives scripts/run_time_varying_parameters_rollout_esmda_from_truth.py,
# which LOADS the ground truth (state.nc + params.nc) from GROUND_TRUTH_DIR
# instead of simulating it, chops it into assimilation windows, and runs
# TimeVaryingParameterESMDA. The number of windows is clamped automatically so
# the rollout never exceeds the time length of the loaded ground truth.
#
# The truth (ground_truth) model is only read here — its forward model is never
# run — so only its solver_name matters (it must match the backend that
# produced the ground truth so the observation grid mapping is correct). The
# ASSIM model is the one actually integrated across the ensemble.
#
# NB: NUM_PARALLEL auto-tracks ENSEMBLE_SIZE but is capped at --cpus-per-task
# (one core per worker), so raise --cpus-per-task to run a larger ensemble fully
# in parallel. Also regenerate the ground truth with the same
# time.output_frequency / time.simulation_time used here so the per-window
# observation vectors align.
#SBATCH --job-name=rollout_from_truth
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=1750M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-rollout_from_truth-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-rollout_from_truth-%j.err

set -euo pipefail

# ============================================================================
# CONFIG — edit these for your run.
# ============================================================================
# ROOT directory holding the pre-simulated ground truth, as written by
# job_scripts/snellius/ground_truth.slurm. The forward-model script nests its
# output under a "<model>_time_varying/" subfolder, so the actual leaf used is
# ${GROUND_TRUTH_DIR}/${GROUND_TRUTH_MODEL}_time_varying (constructed below).
# If you instead point this straight at the leaf (the folder that directly
# contains state.nc + params.nc), set GROUND_TRUTH_SUBDIR="" to disable nesting.
GROUND_TRUTH_DIR="/projects/prjs2075/urbanair/ground_truth"

# Backend that PRODUCED the ground truth (read-only here; sets the obs grid
# mapping) and the backend used to ASSIMILATE.
GROUND_TRUTH_MODEL="pyudales"    # pylbm | pyudales | pypalm
ASSIM_MODEL="pyudales"           # pylbm | pyudales | pypalm

# Geometry case bundle (default domain bounds + sensor layout).
CASE="xie_and_castro"         # xie_and_castro | barcelona

# --- Domain arguments (grid resolution + physical bounds) -------------------
NX=60
NY=40
NZ=16
# Physical bounds as [lo, hi] per axis, in metres: x, y, z.
X_BOUNDS="[-20.0, 80.0]"
Y_BOUNDS="[0.0, 80.0]"
Z_BOUNDS="[0.0, 32.0]"

# --- Observation sensors ----------------------------------------------------
# Sensor coordinates [m], one entry per sensor across the three lists. They
# MUST lie inside the domain AND inside the solver's grid. For uDALES the
# velocity components live on STAGGERED faces whose last node sits one cell
# inside the domain max (e.g. xm only reaches x_max - dx), so a sensor exactly
# at a domain bound (the canonical x=40 layout) falls outside and the
# interpolation aborts. Keep sensors strictly interior — here the downstream
# sensor is pulled to 38.0 (was 40.0) to stay on the uDALES xm axis.
X_POINTS="[20.0, 20.0, 38.0]"
Y_POINTS="[20.0, 10.0, 10.0]"
Z_POINTS="[3.0, 3.0, 3.0]"

# --- Ensemble / ESMDA -------------------------------------------------------
ENSEMBLE_SIZE=96             # number of ensemble members
# Parallel workers. Leave empty to auto-track ENSEMBLE_SIZE (capped at the
# allocated --cpus-per-task); set a number to pin it explicitly.
NUM_PARALLEL=""
NUM_ASSIM_WINDOWS=5          # requested windows (auto-clamped to the truth length)
NUM_TIME_POINTS=3           # time-varying parameter knots per window
NUM_ESMDA_STEPS=2            # ESMDA iterations per window

# --- Time discretisation (MUST match the ground-truth generation) -----------
SIMULATION_TIME=180.0        # per-window length [s]
OUTPUT_FREQUENCY=5.0         # state snapshot interval [s]
SPINUP_TIME=10.0             # constant-inflow plateau before each window [s]

SEED=0
SKIP_VIZ="false"             # set "true" to skip animations/plots
# Correlation-localization truncation; pinned so the update is well-posed at
# small ensemble sizes. Set to "null" (via an extra CLI override) for global.
TRUNCATION_CORRELATION=0.3
# ============================================================================

ENV="snellius"
PROJECT_STORAGE="/projects/prjs2075/urbanair"
# Final outputs land under "assim_from_ground_truth", in a subdir tagged with
# the assim model and the run's domain / ensemble size so different
# configurations don't overwrite each other. NB: two runs with identical
# (assim_model, nx, ny, nz, ensemble_size) reuse the same subdir — append
# something unique (e.g. _${SLURM_JOB_ID}) below if you need to keep both.
RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}"
RESULTS_DIR="${PROJECT_STORAGE}/assim_from_ground_truth/${RUN_TAG}"

# The ground-truth forward run nests its output under "<model>_time_varying/",
# so select the leaf that matches the chosen ground-truth backend. This keeps
# the load unambiguous even when several backends' truths share one root.
# Override with GROUND_TRUTH_SUBDIR (e.g. "" if GROUND_TRUTH_DIR is the leaf).
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
# allocation actually gave us (one worker per core). A larger ENSEMBLE_SIZE than
# cores still runs — the extra members are processed in batches.
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

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Bounds: x=${X_BOUNDS} y=${Y_BOUNDS} z=${Z_BOUNDS}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} windows<=${NUM_ASSIM_WINDOWS} cores=${SLURM_CPUS_PER_TASK}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# One core per ensemble worker; keep BLAS single-threaded so workers don't
# fight for the allocated cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Per-model solver flags for the ASSIM model only — the truth model is read
# from disk, never integrated, so its forward-model config is left untouched.
EXTRA_FLAGS=()
case "${ASSIM_MODEL}" in
  pylbm)
    EXTRA_FLAGS+=( "assim_model.forward_model.cuda=false" )
    # pylbm hardcodes ./.temp relative to cwd; point it at node-local scratch.
    rm -rf .temp
    ln -sfn "${RUN_TEMP_DIR}" .temp
    ;;
  pyudales)
    EXTRA_FLAGS+=( "+assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
    EXTRA_FLAGS+=( "+assim_model.forward_model.output_dir=${RUN_TEMP_DIR}/outputs" )
    ;;
  pypalm)
    EXTRA_FLAGS+=( "+assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
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
    scripts/run_time_varying_parameters_rollout_esmda_from_truth.py \
    model@truth_model="${GROUND_TRUTH_MODEL}" \
    model@assim_model="${ASSIM_MODEL}" \
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
    time_varying.num_time_points="${NUM_TIME_POINTS}" \
    ensemble.ensemble_size="${ENSEMBLE_SIZE}" \
    ensemble.num_parallel_processes="${NUM_PARALLEL}" \
    esmda.num_assimilation_windows="${NUM_ASSIM_WINDOWS}" \
    esmda.num_steps="${NUM_ESMDA_STEPS}" \
    esmda.localization.truncation_correlation="${TRUNCATION_CORRELATION}" \
    esmda.seed="${SEED}" \
    run.ground_truth_dir="${GROUND_TRUTH_PATH}" \
    run.skip_viz="${SKIP_VIZ}" \
    paths.base_results_dir="${PROJECT_STORAGE}" \
    "hydra.run.dir=${RESULTS_DIR}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "Done — rollout ESMDA outputs under ${RESULTS_DIR}"
