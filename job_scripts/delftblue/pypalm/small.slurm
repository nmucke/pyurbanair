#!/bin/bash
#SBATCH --job-name=esmda_pypalm_small
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-esmda_pypalm_small-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-esmda_pypalm_small-%j.err

# Extra args passed to sbatch are forwarded as Hydra overrides, e.g.
#   sbatch $0 time=large esmda.seed=7
# They override this size's group selections. The #SBATCH walltime/cpus above
# are fixed — bump them (or pass `sbatch --time=...`) for a heavier config.

set -euo pipefail

# Run from the repo root regardless of where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

# Logs -> job_scripts/delftblue/out_files; temp/intermediate I/O -> scratch;
# final results -> ${PROJECT_STORAGE}.
PROJECT_STORAGE="/projects/urbanair"
SCRATCH_DIR="/scratch/${USER}"
RUN_TEMP_DIR="${SCRATCH_DIR}/urbanair_temp/${SLURM_JOB_ID}"
mkdir -p job_scripts/delftblue/out_files "${PROJECT_STORAGE}" "${RUN_TEMP_DIR}"

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Repo: $(pwd)"
echo "Results: ${PROJECT_STORAGE}  (temp: ${RUN_TEMP_DIR})"
echo "Extra Hydra overrides: $*"

module purge

# Force ob1/tcp so a stale pixi cache or unsourced activation can't pull
# UCX/InfiniBand into the MPI stack and crash MPI_Finalize.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# Each ensemble worker pins to one core; keep in-rank BLAS single-threaded so
# workers x BLAS threads don't fight for the allocated cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Stderr is fully buffered when redirected to a file — without this a crash
# mid-run loses the traceback (cf. job 9842407).
export PYTHONUNBUFFERED=1

# PALM launches a nested `mpirun palm` per ensemble member. The ensemble runs
# each member in a forkserver worker pinned to a single core (default, tuned
# for uDALES); confined to one CPU, OpenMPI's orted + palm rank with the forced
# TCP BTL livelock and the member run stalls (the serial truth run survives
# only because the main process keeps the full cpuset). Disable pinning so
# members get the whole allocation like the truth run, and let OMPI yield when
# idle / not bind / oversubscribe so concurrent PALMs coexist.
export PYURBANAIR_DISABLE_CPU_PINNING=1
export OMPI_MCA_mpi_yield_when_idle=1
export OMPI_MCA_rmaps_base_oversubscribe=true
export OMPI_MCA_hwloc_base_binding_policy=none

# Route palmrun's per-run working dir (fast_io_catalog) to node-local /tmp
# instead of beegfs scratch. palmrun copies the full build tree (~750 files:
# sources + prebuilt objects + the executable) there per member per window;
# on beegfs that many-small-file copy is slow, on local SSD it's cheap. The
# binary is identical across members, so this is pure I/O placement. Cleaned
# on exit (node-local, so it would vanish with the node anyway).
export PYPALM_FAST_IO_CATALOG="/tmp/urbanair_palm_${SLURM_JOB_ID}"
mkdir -p "${PYPALM_FAST_IO_CATALOG}"
trap 'rm -rf "${PYPALM_FAST_IO_CATALOG}"' EXIT

# The delftblue activation `module load`s nvhpc/25.7, which prepends NVHPC's
# mpif90 (nvfortran) ahead of the conda env's gfortran wrapper. PALM's installer
# then picks nvfortran, which can't read the gfortran-built netcdf.mod and the
# build fails (cf. jobs 9949950 / 9950028). Compute nodes have no internet so
# pypalm's auto-downloader can't recover from a wiped tree — do NOT delete
# palm_model_system/. Instead drop any stale CMakeCache.txt so it can't pin the
# wrong compiler, then force conda's bin (and unset NVHPC's CC/.../OPAL_PREFIX)
# inside the pixi run so install and palmrun use the matching toolchain.
if [ -d libs/pypalm/palm_model_system ]; then
    find libs/pypalm/palm_model_system -type f -name CMakeCache.txt -delete
fi

pixi run -e delftblue -- bash -c '
  unset OPAL_PREFIX CC CXX F77 F90 FC
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  exec python -u scripts/run_time_varying_parameters_rollout_esmda.py \
      size=small \
      domain.nz=16 \
      model@truth_model=pypalm \
      model@assim_model=pypalm \
      ensemble.num_parallel_processes=8 \
      localization=correlation \
      paths.base_results_dir='"${PROJECT_STORAGE}"' \
      +truth_model.forward_model.temp_dir='"${RUN_TEMP_DIR}"' \
      +assim_model.forward_model.temp_dir='"${RUN_TEMP_DIR}"' \
      "$@"
' bash "$@"
