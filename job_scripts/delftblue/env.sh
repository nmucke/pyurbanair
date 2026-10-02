# DelftBlue setup, sourced by every job script here. Submit from the repo root.
# Sets the machine and results root (configs/common.yaml reads them; CLI
# overrides win) and defines `run` (a command in the delftblue pixi env).
set -euo pipefail

module purge

export PYURBANAIR_MACHINE=delftblue PYURBANAIR_RESULTS_ROOT=/projects/urbanair

# The ensemble workers set their own parallelism; keep BLAS single-threaded.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Solver scratch (paths.scratch.delftblue) is per job; kept on failure for
# debugging.
scratch="/scratch/${USER}/urbanair_temp/${SLURM_JOB_ID}"
export PYPALM_FAST_IO_CATALOG="/tmp/urbanair_palm_${SLURM_JOB_ID}"
mkdir -p "${scratch}" "${PYPALM_FAST_IO_CATALOG}"
trap 'status=$?; rm -rf "${PYPALM_FAST_IO_CATALOG}"; [ "${status}" = 0 ] && rm -rf "${scratch}"' EXIT

# The LBM build writes into its own source tree: give each job a private copy
# so concurrent jobs don't clobber each other.
rsync -a --delete --exclude='.git' libs/pylbm/LBM/ "${scratch}/LBM/"
export PYLBM_LBM_PATH="${scratch}/LBM"

# PALM runs a nested `mpirun -n <ncpu>` under --ntasks=1, which counts one
# slot; DelftBlue's OpenMPI 4 still takes the rmaps oversubscribe setting.
export OMPI_MCA_mpi_yield_when_idle=1 OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_oversubscribe=true

if [[ " $* " == *pypalm* ]]; then
    # The activation loads nvhpc, whose mpif90 (nvfortran) shadows conda's
    # gfortran that PALM is built with: drop stale CMake caches and nvhpc's
    # compiler variables.
    find libs/pypalm/palm_model_system -type f -name CMakeCache.txt -delete
    run() {
        pixi run -e delftblue -- bash -c '
            unset OPAL_PREFIX CC CXX F77 F90 FC
            export PATH="${CONDA_PREFIX}/bin:${PATH}"
            exec "$@"' bash "$@"
    }
else
    run() { pixi run -e delftblue -- "$@"; }
fi
