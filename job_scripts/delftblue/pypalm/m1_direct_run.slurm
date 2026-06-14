#!/bin/bash
#SBATCH --job-name=pypalm_m1_direct
#SBATCH --partition=compute-p1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-pypalm_m1_direct-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-pypalm_m1_direct-%j.err

# M1 unit test: drive pypalm.direct_palm.run_direct against the same
# tiny config as M0 and verify u/v/w match the M0 palmrun reference.
# Phase timings are printed so we can compare against the ~134s palmrun baseline.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

SCRATCH_DIR="/scratch/${USER}"
RUN_TEMP_DIR="${SCRATCH_DIR}/m1_direct/${SLURM_JOB_ID}/temp"
M1_FAST_IO="${SCRATCH_DIR}/m1_direct/${SLURM_JOB_ID}/fast_io"
mkdir -p job_scripts/delftblue/out_files "${RUN_TEMP_DIR}" "${M1_FAST_IO}"

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Temp:     ${RUN_TEMP_DIR}"
echo "fast_io:  ${M1_FAST_IO}"

module purge

# Same OMPI safety net as tiny.slurm — direct-run still uses mpirun for PALM
# itself, so the ob1/tcp + yield-when-idle settings still apply.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export PYURBANAIR_DISABLE_CPU_PINNING=1
export OMPI_MCA_mpi_yield_when_idle=1
export OMPI_MCA_rmaps_base_oversubscribe=true
export OMPI_MCA_hwloc_base_binding_policy=none

export PYPALM_FAST_IO_CATALOG="${M1_FAST_IO}"
export M1_TEMP="${RUN_TEMP_DIR}"

# Match tiny.slurm: drop any stale CMakeCache so a build, if it happens, picks
# the right compiler. Direct-run never invokes palmbuild so this is defensive.
if [ -d libs/pypalm/palm_model_system ]; then
    find libs/pypalm/palm_model_system -type f -name CMakeCache.txt -delete
fi

pixi run -e delftblue -- bash -c '
  unset OPAL_PREFIX CC CXX F77 F90 FC
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  exec python -u job_scripts/delftblue/pypalm/m1_direct_run.py
'
