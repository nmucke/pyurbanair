#!/bin/bash
# Generate a ground-truth forward simulation with TIME-VARYING inflow on
# DelftBlue. Unlike the rollout-ESMDA runners (job_scripts/delftblue/<model>/
# rollout_esmda_from_truth.slurm + ../common.sh), this is a SELF-CONTAINED
# submit file: edit the CONFIG block below, then submit it directly:
#
#     sbatch job_scripts/delftblue/ground_truth.slurm
#
# Extra Hydra overrides may also be appended on the command line and take
# precedence over the CONFIG block, e.g.:
#
#     sbatch job_scripts/delftblue/ground_truth.slurm time.simulation_time=240
#
# It drives scripts/run_forward_model.py with params=dynamic_truth, which draws
# the inflow (inflow_angle / velocity_magnitude) from the AR(2) relaxation
# TRUTH sampler (conf/params/dynamic_truth.yaml -- kept separate from the
# assimilation prior `dynamic` to avoid the inverse crime) and runs a single
# forward simulation (optionally rolled out over ROLLOUT_STEPS extra windows).
# The simulated state and the parameter profile are written under ${GT_DIR} as
# <model>_time_varying/{state.nc,params.nc} (the leaf gains a `_rollout` infix
# when ROLLOUT_STEPS > 1; plus animations unless viz is off). That leaf is what
# the rollout-ESMDA runners load via GROUND_TRUTH_DIR/GROUND_TRUTH_SUBDIR in
# job_scripts/delftblue/common.sh.
#
# DelftBlue partitions: compute-p1 nodes have 48 cores / ~185 GB, compute-p2
# nodes 64 cores / ~250 GB (the old combined `compute` partition is drained).
# This single-solver job fits comfortably on a p1 node.
#SBATCH --job-name=ground_truth
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-ground_truth-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-ground_truth-%j.err

set -euo pipefail

# ============================================================================
# CONFIG — edit these for your ground-truth run.
# ============================================================================
# Which CFD backend generates the truth.
MODEL="pypalm"                 # pylbm | pyudales | pypalm
# Geometry case bundle (provides the default domain bounds + sensor layout).
CASE="xie_and_castro"         # xie_and_castro | barcelona

# --- Domain arguments (grid resolution + physical bounds) -------------------
# Grid cell counts per axis.
NX=60
NY=80
NZ=16
# Physical bounds as [lo, hi] per axis, in metres: x, y, z.
X_BOUNDS="[-20.0, 40.0]"
Y_BOUNDS="[0.0, 80.0]"
Z_BOUNDS="[0.0, 32.0]"

# --- Time / parameter resolution --------------------------------------------
SIMULATION_TIME=1500.0        # length of one simulated window [s]
OUTPUT_FREQUENCY=1.0          # state snapshot interval [s]
SPINUP_TIME=50.0              # constant-inflow plateau before the window [s]
NUM_TIME_POINTS=50           # time-varying parameter knots PER WINDOW
SEED=0                        # RNG seed for the AR(2) truth draw
# Extra windows rolled out after the first (run.rollout_steps). 0 -> a single
# window of SIMULATION_TIME; N -> N+1 windows stitched onto one time axis, with
# the AR(2) params extrapolated window-to-window. NOTE: any value > 1 inserts
# `_rollout` into the output leaf name (see header).
ROLLOUT_STEPS=0

# --- Solver parallelism ------------------------------------------------------
# Number of CPU cores the solver runs on (pyudales/pypalm only; ignored by
# pylbm). Keep this <= --cpus-per-task above.
NCPU=16

SKIP_VIZ="false"              # set "true" to skip animations/plots
# ============================================================================

# DelftBlue is CPU-only on the compute partitions; pixi env "delftblue".
ENV="delftblue"
PROJECT_STORAGE="/projects/urbanair"
# Final outputs land here. resolve_output_dir() in the script honours
# hydra.run.dir, so all outputs are rooted at this "ground_truth" folder.
GT_DIR="${PROJECT_STORAGE}/ground_truth"

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/delftblue/out_files "${GT_DIR}"

# Heavy intermediate solver I/O goes on /scratch (beegfs), not the home quota /
# project space. Cleaned up on success; left behind on failure for inspection.
RUN_TEMP_DIR="/scratch/${USER}/urbanair_temp/${SLURM_JOB_ID}"
mkdir -p "${RUN_TEMP_DIR}"
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Model=${MODEL} case=${CASE} domain=${NX}x${NY}x${NZ} cores=${SLURM_CPUS_PER_TASK} ncpu=${NCPU}"
echo "Bounds: x=${X_BOUNDS} y=${Y_BOUNDS} z=${Z_BOUNDS}"
echo "Window=${SIMULATION_TIME}s knots/window=${NUM_TIME_POINTS} rollout_steps=${ROLLOUT_STEPS}"
echo "Output root: ${GT_DIR}  (temp: ${RUN_TEMP_DIR})"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# DelftBlue MPI environment: force ob1/tcp so a stale pixi cache or unsourced
# activation can't pull UCX/InfiniBand into the MPI stack and crash
# MPI_Finalize. Unlike Snellius, DelftBlue's OpenMPI DOES need osc=pt2pt.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# Single forward simulation, but keep BLAS single-threaded so the solver
# controls its own parallelism rather than fighting nested BLAS threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Stderr is fully buffered to a file — without this a crash loses the traceback.
export PYTHONUNBUFFERED=1

# Per-model solver flags + scratch redirection. Every backend's forward_model
# takes temp_dir directly (it defaults to ${paths.experiment_dir}).
EXTRA_FLAGS=()
case "${MODEL}" in
  pylbm)
    # CPU build on the compute partitions. The LBM Fortran build mutates its own
    # source tree, so give this job a private copy on scratch and point pylbm at
    # it via PYLBM_LBM_PATH (read in libs/pylbm/.../__init__.py).
    JOB_LBM_DIR="${RUN_TEMP_DIR}/LBM"
    rsync -a --delete --exclude='.git' libs/pylbm/LBM/ "${JOB_LBM_DIR}/"
    export PYLBM_LBM_PATH="${JOB_LBM_DIR}"
    EXTRA_FLAGS+=( "model.forward_model.cuda=false" )
    EXTRA_FLAGS+=( "model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
    ;;
  pyudales)
    EXTRA_FLAGS+=( "model.forward_model.ncpu=${NCPU}" )
    EXTRA_FLAGS+=( "model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
    EXTRA_FLAGS+=( "+model.forward_model.output_dir=${RUN_TEMP_DIR}/outputs" )
    ;;
  pypalm)
    EXTRA_FLAGS+=( "model.forward_model.ncpu=${NCPU}" )
    EXTRA_FLAGS+=( "model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
    # PALM launches a nested `mpirun -n ${NCPU} palm`. Under --ntasks=1 OpenMPI
    # counts one SLURM "slot", so tell it to oversubscribe (DelftBlue runs
    # OpenMPI <5, where the rmaps env var still works — no PRRTE flags needed).
    # NCPU <= cpus-per-task, so the cores aren't actually oversubscribed.
    export PYURBANAIR_DISABLE_CPU_PINNING=1
    export OMPI_MCA_mpi_yield_when_idle=1
    export OMPI_MCA_rmaps_base_oversubscribe=true
    export OMPI_MCA_hwloc_base_binding_policy=none
    export PYPALM_USE_DIRECT_RUN="${PYPALM_USE_DIRECT_RUN:-1}"
    # PALM's fast-IO catalog goes to node-local /tmp.
    export PYPALM_FAST_IO_CATALOG="/tmp/urbanair_palm_${SLURM_JOB_ID}"
    mkdir -p "${PYPALM_FAST_IO_CATALOG}"
    # The delftblue activation `module load`s nvhpc, whose mpif90 (nvfortran)
    # shadows the conda env's gfortran wrapper; drop any stale CMakeCache.txt so
    # it can't pin the wrong compiler (the bash -c wrapper below scrubs the rest).
    if [ -d libs/pypalm/palm_model_system ]; then
      find libs/pypalm/palm_model_system -type f -name CMakeCache.txt -delete
    fi
    ;;
  *)
    echo "unsupported MODEL '${MODEL}' (expected pylbm|pyudales|pypalm)" >&2
    exit 1
    ;;
esac

# params=dynamic_truth swaps in the time-varying TRUTH sampler (the old
# run.time_varying=true switch); params.seed seeds the AR(2) draw and the
# window-to-window extrapolation; params.time_coords.num sets the knots per
# window (formerly time_varying.num_time_points).
PY_ARGS=(
  python -u scripts/run_forward_model.py
  model="${MODEL}"
  case="${CASE}"
  params=dynamic_truth
  params.seed="${SEED}"
  params.time_coords.num="${NUM_TIME_POINTS}"
  run.rollout_steps="${ROLLOUT_STEPS}"
  domain.nx="${NX}"
  domain.ny="${NY}"
  domain.nz="${NZ}"
  "domain.bounds=[${X_BOUNDS}, ${Y_BOUNDS}, ${Z_BOUNDS}]"
  time.simulation_time="${SIMULATION_TIME}"
  time.output_frequency="${OUTPUT_FREQUENCY}"
  time.spinup_time="${SPINUP_TIME}"
  run.skip_viz="${SKIP_VIZ}"
  "hydra.run.dir=${GT_DIR}"
  "${EXTRA_FLAGS[@]}"
  "$@"
)

if [ "${MODEL}" = "pypalm" ]; then
  # Route through a bash -c that scrubs nvhpc's CC/F90/... and prepends
  # CONDA_PREFIX/bin so palmbuild + palmrun pick conda's gfortran.
  pixi run -e "${ENV}" -- bash -c '
    unset OPAL_PREFIX CC CXX F77 F90 FC
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    exec "$@"
  ' bash "${PY_ARGS[@]}"
else
  pixi run -e "${ENV}" -- "${PY_ARGS[@]}"
fi

echo "Done — ground-truth outputs under ${GT_DIR}"
