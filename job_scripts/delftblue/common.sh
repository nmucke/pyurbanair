#!/bin/bash
# Shared defaults for the DELFTBLUE rollout-ESMDA-from-truth runners (pyudales,
# pylbm, pypalm). SOURCE this from a backend .slurm runner -- it is not meant to
# be run on its own:
#
#     source "${REPO_ROOT}/job_scripts/delftblue/common.sh"
#
# Everything here is run config that is IDENTICAL across the three backends, so
# the runs stay directly comparable (same ground truth, domain, windows, time
# horizon and dynamic-parameter settings -- only the assimilation solver
# differs). Backend-specific knobs (pixi env, parallelism, solver flags) live in
# each runner; the SWEEP parameters (grid resolution NX/NY/NZ, ENSEMBLE_SIZE,
# NUM_ESMDA_STEPS) also live in the runners so the sweep launchers can inject one
# value per job.
#
# This is the DelftBlue sibling of job_scripts/local/common.sh and carries the
# SAME experiment config; only the paths (project storage + /scratch), the MPI
# environment, and the absence of a parallel-worker cap differ (on DelftBlue the
# SLURM allocation requests one core per ensemble member, capped at a single
# 64-core compute node, and num_parallel == ensemble size).
#
# Every value is env-overridable: `export VAR=...` before invoking a runner or
# sweep launcher changes it for that run. To retune the whole DelftBlue suite at
# once, edit the defaults below.

# --- Paths ------------------------------------------------------------------
# Final per-run outputs land on the project space; heavy intermediate solver I/O
# ("scratch") goes to /scratch/$USER (beegfs). Each job gets its own RUN_TEMP_DIR
# subdir (keyed by SLURM_JOB_ID in the runner), cleaned up on success.
RESULTS_ROOT="${RESULTS_ROOT:-/projects/urbanair/assim_from_ground_truth}"
TEMP_ROOT="${TEMP_ROOT:-/scratch/${USER}/urbanair_temp}"

# ROOT directory holding the pre-simulated ground truth shared by all backends.
# The leaf actually loaded is ${GROUND_TRUTH_DIR}/${GROUND_TRUTH_MODEL}_time_varying
# (set GROUND_TRUTH_SUBDIR="" if GROUND_TRUTH_DIR already points at the leaf).
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-/projects/urbanair/ground_truth_small}"
GROUND_TRUTH_MODEL="${GROUND_TRUTH_MODEL:-pyudales}"   # pylbm | pyudales | pypalm

# --- Geometry / domain size -------------------------------------------------
# Physical domain bounds [lo, hi] per axis in metres and the sensor layout. NOTE:
# the grid resolution NX/NY/NZ (the number of discrete points) is NOT here -- it
# is a sweep parameter and lives in each runner.
CASE="${CASE:-xie_and_castro}"        # xie_and_castro | barcelona
X_BOUNDS="${X_BOUNDS:-[-20.0, 80.0]}"
Y_BOUNDS="${Y_BOUNDS:-[0.0, 80.0]}"
Z_BOUNDS="${Z_BOUNDS:-[0.0, 32.0]}"
# Observation sensors: one entry per sensor across the three lists.
X_POINTS="${X_POINTS:-[30.0, 60.0, 40.0, 10.0, 65.0]}"
Y_POINTS="${Y_POINTS:-[10.0, 20.0, 40.0, 60.0, 50.0]}"
Z_POINTS="${Z_POINTS:-[2.0, 2.0, 2.0, 2.0, 2.0]}"

# --- Assimilation windows ---------------------------------------------------
# Number of assimilation windows the loaded ground truth is chopped into. Keep
# SIMULATION_TIME * NUM_ASSIM_WINDOWS <= the time length of the ground truth.
NUM_ASSIM_WINDOWS="${NUM_ASSIM_WINDOWS:-4}"

# --- Time horizon -----------------------------------------------------------
SIMULATION_TIME="${SIMULATION_TIME:-300.0}"   # per-window length [s]
OUTPUT_FREQUENCY="${OUTPUT_FREQUENCY:-1.0}"    # state snapshot interval [s]
SPINUP_TIME="${SPINUP_TIME:-50.0}"             # constant-inflow plateau before each window [s]

# --- Dynamic (time-varying) parameter settings ------------------------------
# The smoother and parameter groups that make the inflow parameters time-varying,
# plus the number of knots per window (NUM_TIME_POINTS sets both the prior and the
# truth parameterisation). Forwarded verbatim into the run via DYNAMIC_PARAM_FLAGS.
NUM_TIME_POINTS="${NUM_TIME_POINTS:-10}"
DYNAMIC_PARAM_FLAGS=(
  "esmda/smoother=dynamic"
  "params@truth_params=dynamic_truth"
  "params@prior_params=dynamic"
  "prior_params.time_coords.num=${NUM_TIME_POINTS}"
  "params.time_coords.num=${NUM_TIME_POINTS}"
)

# --- Misc run defaults ------------------------------------------------------
SEED="${SEED:-0}"
SKIP_VIZ="${SKIP_VIZ:-false}"          # set "true" to skip animations/plots

# --- Correlation localization -----------------------------------------------
# Shared default is the global (unlocalized) update. Set USE_LOCALIZATION=true to
# enable the localized Kalman update with TRUNCATION_CORRELATION. This resolves
# into LOCALIZATION_FLAGS (Hydra overrides) + LOCALIZATION_TAG (output-dir suffix)
# for the runners to consume.
USE_LOCALIZATION="${USE_LOCALIZATION:-false}"
TRUNCATION_CORRELATION="${TRUNCATION_CORRELATION:-0.3}"
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

# --- Ground-truth leaf resolution + validation ------------------------------
# Resolve the leaf that matches the chosen truth backend and verify it carries
# the pre-simulated state.nc + params.nc. Sets GROUND_TRUTH_PATH for the runners.
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

# --- Cluster environment ----------------------------------------------------
# Force ob1/tcp so a stale pixi cache or unsourced activation can't pull
# UCX/InfiniBand into the MPI stack and crash MPI_Finalize on DelftBlue. Unlike
# Snellius, DelftBlue's OpenMPI DOES need osc=pt2pt. This is the one MPI
# difference between the two clusters.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# Keep BLAS single-threaded so parallel ensemble workers (one per allocated core)
# don't oversubscribe; line-buffer Python output so a crash mid-run keeps its
# traceback.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- The single source of truth for every shared run argument ---------------
# COMMON_RUN_FLAGS is EVERY scripts/run_esmda.py Hydra override that is identical
# across the three backends. Each runner expands it verbatim and only adds the
# bits that genuinely differ: the assimilation model, the per-run sweep values
# (domain.nx/ny/nz, ensemble.ensemble_size, ensemble.num_parallel_processes,
# esmda.num_steps), hydra.run.dir, and any backend-specific solver flags. Defining
# it once here is what guarantees all three models run the EXACT same experiment.
COMMON_RUN_FLAGS=(
  "${DYNAMIC_PARAM_FLAGS[@]}"
  model@truth_model="${GROUND_TRUTH_MODEL}"
  case="${CASE}"
  "domain.bounds=[${X_BOUNDS}, ${Y_BOUNDS}, ${Z_BOUNDS}]"
  obs.mode=points
  "obs.x_points=${X_POINTS}"
  "obs.y_points=${Y_POINTS}"
  "obs.z_points=${Z_POINTS}"
  time.simulation_time="${SIMULATION_TIME}"
  time.output_frequency="${OUTPUT_FREQUENCY}"
  time.spinup_time="${SPINUP_TIME}"
  esmda.num_assimilation_windows="${NUM_ASSIM_WINDOWS}"
  esmda.seed="${SEED}"
  run.truth_dir="${GROUND_TRUTH_PATH}"
  run.skip_viz="${SKIP_VIZ}"
  "${LOCALIZATION_FLAGS[@]}"
)
