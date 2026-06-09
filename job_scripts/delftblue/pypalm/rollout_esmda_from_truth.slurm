#!/bin/bash
# DELFTBLUE rollout-ESMDA-from-truth runner -- pypalm backend (CPU).
#
# DelftBlue sibling of job_scripts/local/pypalm/rollout_esmda_from_truth.sh:
# it submits scripts/run_esmda.py in its loaded-truth mode (run.truth_dir).
# PALM launches a nested `mpirun palm` per ensemble member; the ensemble fans out
# across parallel processes with CPU pinning off and OpenMPI told to
# yield/no-bind/oversubscribe so the concurrent PALMs coexist. Run config shared
# with the pyudales/pylbm runners lives in ../common.sh (sourced below); only the
# pypalm specifics are set here.
#
# Submit it directly (edit ../common.sh to retune the shared config), or append
# Hydra overrides which take precedence:
#
#     sbatch job_scripts/delftblue/pypalm/rollout_esmda_from_truth.slurm
#     sbatch job_scripts/delftblue/pypalm/rollout_esmda_from_truth.slurm esmda.num_steps=4
#
# NX/NY/NZ, ENSEMBLE_SIZE and NUM_ESMDA_STEPS are read from the environment (the
# sweep launchers in this folder inject one value per job via --export), so each
# configuration lands in its own RESULTS_DIR.
#
# No parallel cap: num_parallel == ensemble size; the sweep launchers request one
# core per ensemble member (--cpus-per-task == ENSEMBLE_SIZE, capped at a 64-core
# node -- larger ensembles oversubscribe, which PALM tolerates).
#
# NB: direct-run (the default) reuses the prebuilt PALM binary at
# libs/pypalm/palm_model_system/MAKE_DEPOSITORY_default/{palm,combine_plot_fields.x};
# compute nodes have no internet, so build it once on a login node before
# submitting. Prefix with `PYPALM_USE_DIRECT_RUN=` to fall back to palmrun/palmbuild.
#SBATCH --job-name=rollout_pypalm
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --time=16:00:00
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-rollout_pypalm-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-rollout_pypalm-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Backend-specific knobs ------------------------------------------------------
ENV="delftblue"
ASSIM_MODEL="pypalm"

# Sweep parameters (grid resolution / ensemble / ESMDA steps). Env-overridable so
# the sweep launchers can inject one value per job; each lands in its own RESULTS_DIR.
NX="${NX:-50}"
NY="${NY:-40}"
NZ="${NZ:-16}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-96}"
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-20.0}"   # obs.interval_seconds: time-aggregation bin width [s]

# PALM cannot run with fewer than 16 vertical levels; enforce the floor so a
# shared sweep handing pypalm a coarser nz (e.g. the k=1 domain-sweep row) is
# raised to 16 rather than crashing the solver. Done before RUN_TAG so the run
# still lands in its own nz16 output dir.
if (( NZ < 16 )); then
  echo "note: pypalm requires nz>=16; raising NZ ${NZ} -> 16" >&2
  NZ=16
fi

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

# PALM's fast-IO catalog goes to node-local /tmp (cheap symlink/copy staging
# under direct-run; load-bearing under the palmrun fallback). Cleaned up with the
# scratch temp dir at exit.
PYPALM_FAST_IO_CATALOG="/tmp/urbanair_palm_${SLURM_JOB_ID:-$$}"

mkdir -p job_scripts/delftblue/out_files "${RESULTS_DIR}" "${RUN_TEMP_DIR}" "${PYPALM_FAST_IO_CATALOG}"
# Clean this run's scratch on success (and the node-local fast-IO catalog always);
# leave RUN_TEMP_DIR for debugging on failure.
trap 'rm -rf "${PYPALM_FAST_IO_CATALOG}"; [ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "DELFTBLUE pypalm rollout-ESMDA: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} parallel=${NUM_PARALLEL} windows=${NUM_ASSIM_WINDOWS} cores=${SLURM_CPUS_PER_TASK:-?}"
echo "ESMDA steps=${NUM_ESMDA_STEPS} obs_interval=${INTERVAL_SECONDS}s localization=${USE_LOCALIZATION}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# PALM-specific environment: let the nested per-member mpirun ranks coexist with
# the unpinned ensemble workers. (The base ob1/tcp + osc=pt2pt MPI settings come
# from common.sh.)
export PYURBANAIR_DISABLE_CPU_PINNING=1
export OMPI_MCA_mpi_yield_when_idle=1
export OMPI_MCA_rmaps_base_oversubscribe=true
export OMPI_MCA_hwloc_base_binding_policy=none
export PYPALM_USE_DIRECT_RUN="${PYPALM_USE_DIRECT_RUN:-1}"
export PYPALM_FAST_IO_CATALOG

# The delftblue activation `module load`s nvhpc, whose mpif90 (nvfortran) shadows
# the conda env's gfortran wrapper. PALM's installer then picks nvfortran, which
# can't read the gfortran-built netcdf.mod and the build fails. Drop any stale
# CMakeCache.txt so it can't pin the wrong compiler; the inner bash wrapper below
# forces conda's bin (and unsets NVHPC's CC/... env) so the matching toolchain.
if [ -d libs/pypalm/palm_model_system ]; then
  find libs/pypalm/palm_model_system -type f -name CMakeCache.txt -delete
fi

# pypalm scratch lands under this job's private temp dir.
EXTRA_FLAGS=(
  "assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}"
)

# COMMON_RUN_FLAGS (from common.sh) carries every shared Hydra override; only the
# assim model, the per-run sweep values, hydra.run.dir and the pypalm solver flags
# are added here. The run is routed through a bash -c that scrubs nvhpc's CC/F90/...
# and prepends CONDA_PREFIX/bin so palmbuild + palmrun pick conda's gfortran.
PY_ARGS=(
  python -u scripts/run_esmda.py
  "${COMMON_RUN_FLAGS[@]}"
  model@assim_model="${ASSIM_MODEL}"
  domain.nx="${NX}"
  domain.ny="${NY}"
  domain.nz="${NZ}"
  ensemble.ensemble_size="${ENSEMBLE_SIZE}"
  ensemble.num_parallel_processes="${NUM_PARALLEL}"
  esmda.num_steps="${NUM_ESMDA_STEPS}"
  obs.interval_seconds="${INTERVAL_SECONDS}"
  "hydra.run.dir=${RESULTS_DIR}"
  "${EXTRA_FLAGS[@]}"
  "$@"
)

pixi run -e "${ENV}" -- bash -c '
  unset OPAL_PREFIX CC CXX F77 F90 FC
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  exec "$@"
' bash "${PY_ARGS[@]}"

echo "Done -- rollout ESMDA outputs under ${RESULTS_DIR}"
