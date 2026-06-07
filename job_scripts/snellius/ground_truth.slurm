#!/bin/bash
# Generate a ground-truth forward simulation with TIME-VARYING inflow on
# Snellius. Unlike the ESMDA template (job_scripts/snellius/submit.sh +
# templates/esmda.slurm), this is a SELF-CONTAINED submit file: edit the
# CONFIG block below, then submit it directly:
#
#     sbatch job_scripts/snellius/ground_truth.slurm
#
# Extra Hydra overrides may also be appended on the command line and take
# precedence over the CONFIG block, e.g.:
#
#     sbatch job_scripts/snellius/ground_truth.slurm time.simulation_time=240
#
# It drives scripts/run_forward_model.py with run.time_varying=true, which draws
# the inflow (inflow_angle / velocity_magnitude) from the external-prior AR(2)
# relaxation truth model and runs a single forward simulation. The simulated
# state and the parameter profile are written under
# ${PROJECT_STORAGE}/ground_truth/ as
# <model>_time_varying/{state.nc,params.nc} (plus animations unless viz is off).
#SBATCH --job-name=ground_truth
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=1750M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-ground_truth-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-ground_truth-%j.err

set -euo pipefail

# ============================================================================
# CONFIG — edit these for your ground-truth run.
# ============================================================================
# Which CFD backend generates the truth.
MODEL="pyudales"                 # pylbm | pyudales | pypalm
# Geometry case bundle (provides the default domain bounds + sensor layout).
CASE="xie_and_castro"         # xie_and_castro | barcelona

# --- Domain arguments (grid resolution + physical bounds) -------------------
# Grid cell counts per axis.
NX=200
NY=160
NZ=64
# Physical bounds as [lo, hi] per axis, in metres: x, y, z.
X_BOUNDS="[-20.0, 80.0]"
Y_BOUNDS="[0.0, 80.0]"
Z_BOUNDS="[0.0, 32.0]"

# --- Time / parameter resolution --------------------------------------------
SIMULATION_TIME=3600.0         # length of the simulated window [s]
OUTPUT_FREQUENCY=5.0          # state snapshot interval [s]
SPINUP_TIME=10.0              # constant-inflow plateau before the window [s]
NUM_TIME_POINTS=240            # number of time-varying parameter knots
SEED=0                        # RNG seed for the AR(2) truth draw

SKIP_VIZ="false"              # set "true" to skip animations/plots
# ============================================================================

# Snellius is CPU-only here; pixi env "snellius" sets JAX_PLATFORMS=cpu.
ENV="snellius"
PROJECT_STORAGE="/projects/prjs2075/urbanair"
# Final outputs land here. resolve_output_dir() in the script honours
# hydra.run.dir, so all outputs are rooted at this "ground_truth" folder.
GT_DIR="${PROJECT_STORAGE}/ground_truth"

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/snellius/out_files "${GT_DIR}"

# Heavy intermediate solver I/O goes on /scratch-shared, not the home quota /
# project space. Cleaned up on success; left behind on failure for inspection.
RUN_TEMP_DIR="/scratch-shared/${USER}/urbanair_temp/${SLURM_JOB_ID}"
mkdir -p "${RUN_TEMP_DIR}"
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Model=${MODEL} case=${CASE} domain=${NX}x${NY}x${NZ} cores=${SLURM_CPUS_PER_TASK}"
echo "Bounds: x=${X_BOUNDS} y=${Y_BOUNDS} z=${Z_BOUNDS}"
echo "Output root: ${GT_DIR}  (temp: ${RUN_TEMP_DIR})"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# Single forward simulation, but keep BLAS single-threaded so the solver
# controls its own parallelism rather than fighting nested BLAS threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Stderr is fully buffered to a file — without this a crash loses the traceback.
export PYTHONUNBUFFERED=1

# Per-model solver flags + scratch redirection.
EXTRA_FLAGS=()
case "${MODEL}" in
  pylbm)
    # CPU build on the compute partition.
    EXTRA_FLAGS+=( "model.forward_model.cuda=false" )
    # pylbm hardcodes ./.temp relative to cwd; point it at node-local scratch.
    rm -rf .temp
    ln -sfn "${RUN_TEMP_DIR}" .temp
    ;;
  pyudales)
    EXTRA_FLAGS+=( "+model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
    EXTRA_FLAGS+=( "+model.forward_model.output_dir=${RUN_TEMP_DIR}/outputs" )
    ;;
  pypalm)
    EXTRA_FLAGS+=( "+model.forward_model.temp_dir=${RUN_TEMP_DIR}" )
    export PYURBANAIR_DISABLE_CPU_PINNING=1
    export PYPALM_USE_DIRECT_RUN="${PYPALM_USE_DIRECT_RUN:-1}"
    export PYPALM_FAST_IO_CATALOG="${TMPDIR}/urbanair_palm_${SLURM_JOB_ID}"
    mkdir -p "${PYPALM_FAST_IO_CATALOG}"
    ;;
  *)
    echo "unsupported MODEL '${MODEL}' (expected pylbm|pyudales|pypalm)" >&2
    exit 1
    ;;
esac

pixi run -e "${ENV}" -- python -u scripts/run_forward_model.py \
    model="${MODEL}" \
    run.time_varying=true \
    case="${CASE}" \
    domain.nx="${NX}" \
    domain.ny="${NY}" \
    domain.nz="${NZ}" \
    "domain.bounds=[${X_BOUNDS}, ${Y_BOUNDS}, ${Z_BOUNDS}]" \
    time.simulation_time="${SIMULATION_TIME}" \
    time.output_frequency="${OUTPUT_FREQUENCY}" \
    time.spinup_time="${SPINUP_TIME}" \
    time_varying.num_time_points="${NUM_TIME_POINTS}" \
    esmda.seed="${SEED}" \
    run.skip_viz="${SKIP_VIZ}" \
    paths.base_results_dir="${PROJECT_STORAGE}" \
    "hydra.run.dir=${GT_DIR}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "Done — ground-truth outputs under ${GT_DIR}"
