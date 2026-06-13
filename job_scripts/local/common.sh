#!/bin/bash
# Shared defaults for the LOCAL rollout-ESMDA-from-truth runners (pyudales,
# pylbm, pypalm). SOURCE this from a backend runner -- it is not executable on
# its own:
#
#     source "${REPO_ROOT}/job_scripts/local/common.sh"
#
# Everything here is run config that is IDENTICAL across the three backends, so
# the runs stay directly comparable (same ground truth, domain, windows, time
# horizon and dynamic-parameter settings -- only the assimilation solver
# differs). Backend-specific knobs (pixi env, GPU vs CPU, parallelism policy,
# solver flags) live in each runner; the SWEEP parameters (grid resolution
# NX/NY/NZ, ENSEMBLE_SIZE, NUM_ESMDA_STEPS, INTERVAL_SECONDS) also live in the
# runners so the sweep launchers can inject one value per job.
#
# Every value is env-overridable: `export VAR=...` before invoking a runner or
# sweep launcher changes it for that run. To retune the whole local suite at
# once, edit the defaults below.

# Requires REPO_ROOT (the runner sets it before sourcing) to anchor the default
# local paths under the repo.
: "${REPO_ROOT:?common.sh: REPO_ROOT must be set before sourcing}"

# --- Paths ------------------------------------------------------------------
# Local run -> outputs and heavy intermediate solver I/O ("scratch") stay under
# the repo by default. RESULTS_ROOT holds final per-run outputs; TEMP_ROOT is the
# scratch base (each run gets its own RUN_TAG_$$ subdir, cleaned up on success).
# RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/assim_from_ground_truth}"
RESULTS_ROOT="/export/scratch2/ntm/results/assim_from_ground_truth"
TEMP_ROOT="${TEMP_ROOT:-${REPO_ROOT}/.local_runs/temp}"

# Directory holding the pre-simulated ground truth shared by all backends -- the
# local copy of the Snellius truth (snellius:/projects/prjs2075/urbanair/
# ground_truth_pyudales_wide/pyudales_time_varying): 60x80x16 on x=[-20,40],
# y=[0,80], z=[0,32], 1500 s. Here state.nc + params.nc sit directly in
# GROUND_TRUTH_DIR, so the default leaf subdir is empty (set GROUND_TRUTH_SUBDIR
# to e.g. pyudales_time_varying if you point GROUND_TRUTH_DIR at a root that
# nests per-model leaves, as on Snellius).
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-/export/scratch1/ntm/pyurbanair/ground_truth}"
GROUND_TRUTH_MODEL="${GROUND_TRUTH_MODEL:-pyudales}"   # pylbm | pyudales | pypalm

# --- Geometry / domain size -------------------------------------------------
# Physical domain bounds [lo, hi] per axis in metres and the sensor layout. NOTE:
# the grid resolution NX/NY/NZ (the number of discrete points) is NOT here -- it
# is a sweep parameter and lives in each runner.
CASE="${CASE:-xie_and_castro}"        # xie_and_castro | barcelona
X_BOUNDS="${X_BOUNDS:-[-20.0, 40.0]}"
Y_BOUNDS="${Y_BOUNDS:-[0.0, 80.0]}"
Z_BOUNDS="${Z_BOUNDS:-[0.0, 32.0]}"
# Observation sensors: one entry per sensor across the three lists.
X_POINTS="${X_POINTS:-[10.0, 10.0, 20.0, 20.0, 30.0, 30.0]}"
Y_POINTS="${Y_POINTS:-[20.0, 60.0, 10.0, 50.0, 30.0, 70.0]}"
Z_POINTS="${Z_POINTS:-[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]}"

# --- Assimilation windows ---------------------------------------------------
# Number of assimilation windows the loaded ground truth is chopped into. Keep
# SIMULATION_TIME * NUM_ASSIM_WINDOWS <= the time length of the ground truth.
NUM_ASSIM_WINDOWS="${NUM_ASSIM_WINDOWS:-6}"

# --- Time horizon -----------------------------------------------------------
SIMULATION_TIME="${SIMULATION_TIME:-180.0}"   # per-window length [s]
OUTPUT_FREQUENCY="${OUTPUT_FREQUENCY:-2.0}"    # state snapshot interval [s]
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

# Maximum number of parallel ensemble processes a CPU-backend run (pyudales,
# pypalm) may use -- CHOOSE this. The actual worker count is
# min(ENSEMBLE_SIZE, LOCAL_MAX_PARALLEL), so it never exceeds your chosen ceiling
# and never spawns more workers than there are ensemble members. Independent of
# the machine's core count (set it to your core budget). Runs are sequential (one
# after another, no scheduler). pylbm ignores this (it is GPU, single-process).
# Choose it here, per run (`LOCAL_MAX_PARALLEL=32 bash …`), or pin an exact worker
# count with NUM_PARALLEL=….
LOCAL_MAX_PARALLEL="${LOCAL_MAX_PARALLEL:-16}"

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
GROUND_TRUTH_SUBDIR="${GROUND_TRUTH_SUBDIR-}"
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

# Keep BLAS single-threaded so parallel ensemble workers don't oversubscribe the
# cores; line-buffer Python output so a crash mid-run keeps its traceback.
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
