#!/bin/bash
#SBATCH --job-name=pypalm_m0_capture
#SBATCH --partition=compute-p1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-pypalm_m0_capture-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-pypalm_m0_capture-%j.err

# M0 capture: run pypalm tiny twice (combine on, combine off), keep palmrun's
# tempdirs around (palmrun -B), then run combine_plot_fields.x bare (no mpirun)
# on the second run's tempdir. Stash everything under
# /scratch/$USER/m0_capture/$SLURM_JOB_ID/ for offline diffing on the login node.
#
# Differences from tiny.slurm:
#  - PYPALM_FAST_IO_CATALOG points at /scratch (NOT /tmp) and has NO EXIT trap,
#    so palmrun's tempdirs survive past job end and can be inspected.
#  - Runs job_scripts/delftblue/pypalm/m0_capture.py instead of the ESMDA driver.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

SCRATCH_DIR="/scratch/${USER}"
RUN_TEMP_DIR="${SCRATCH_DIR}/m0_capture/${SLURM_JOB_ID}/temp"
M0_STASH="${SCRATCH_DIR}/m0_capture/${SLURM_JOB_ID}/stash"
M0_FAST_IO="${SCRATCH_DIR}/m0_capture/${SLURM_JOB_ID}/fast_io"
mkdir -p job_scripts/delftblue/out_files "${RUN_TEMP_DIR}" "${M0_STASH}" "${M0_FAST_IO}"

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Stash:    ${M0_STASH}"
echo "Temp:     ${RUN_TEMP_DIR}"
echo "fast_io:  ${M0_FAST_IO}"

module purge

# Same OMPI safety net as tiny.slurm.
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

# palmrun tempdir lives under /scratch with NO trap — survives past job exit.
export PYPALM_FAST_IO_CATALOG="${M0_FAST_IO}"

# Mirror tiny.slurm's CMakeCache workaround in case a build is triggered.
if [ -d libs/pypalm/palm_model_system ]; then
    find libs/pypalm/palm_model_system -type f -name CMakeCache.txt -delete
fi

export M0_TEMP="${RUN_TEMP_DIR}"
export M0_STASH

pixi run -e delftblue -- bash -c '
  unset OPAL_PREFIX CC CXX F77 F90 FC
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  exec python -u job_scripts/delftblue/pypalm/m0_capture.py
'

echo "Stash contents at job end:"
find "${M0_STASH}" -maxdepth 3 -printf "%p\n" | head -100
