# Snellius setup, sourced by every job script here. Submit from the repo root.
# Sets the machine and results root (configs/common.yaml reads them; CLI
# overrides win) and defines `run` (a command in the snellius pixi env).
set -euo pipefail

module purge

export PYURBANAIR_MACHINE=snellius PYURBANAIR_RESULTS_ROOT=/projects/prjs2075/urbanair

# The ensemble workers set their own parallelism; keep BLAS single-threaded.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Solver scratch (paths.scratch.snellius) is per job on /scratch-shared, not
# $TMPDIR: pyudales' write_inputs.sh fails under /scratch-local. Kept on
# failure for debugging.
trap '[ "$?" = 0 ] && rm -rf "/scratch-shared/${USER}/urbanair_temp/${SLURM_JOB_ID}"' EXIT

# PALM runs a nested `mpirun -n <ncpu>` under --ntasks=1. OpenMPI 5 (PRRTE)
# counts one slot, so it needs --map-by :OVERSUBSCRIBE; ncpu <= cpus-per-task,
# so cores aren't really oversubscribed.
export OMPI_MCA_mpi_yield_when_idle=1 OMPI_MCA_hwloc_base_binding_policy=none
export PYPALM_MPIRUN_EXTRA_ARGS="--map-by :OVERSUBSCRIBE"
export PYPALM_FAST_IO_CATALOG="${TMPDIR}/urbanair_palm_${SLURM_JOB_ID}"
mkdir -p "${PYPALM_FAST_IO_CATALOG}"

run() { pixi run -e snellius -- "$@"; }
