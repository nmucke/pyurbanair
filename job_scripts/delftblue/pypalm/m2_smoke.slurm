#!/bin/bash
#SBATCH --job-name=pypalm_m2_smoke
#SBATCH --partition=compute
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-pypalm_m2_smoke-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-pypalm_m2_smoke-%j.err

# M2 smoke test: drive the full ESMDA rollout pipeline under
# PYPALM_USE_DIRECT_RUN=1 with a minimal config (1 ensemble member, 1 window,
# 1 ESMDA step) so we exercise every code path (truth + ensemble + worker pool)
# but stay short enough to fail fast on any wiring issue. Roughly 4 PALM
# invocations vs the 21 that tiny.slurm makes.
#
# Validates:
#  - ForwardModel.run() with PYPALM_USE_DIRECT_RUN=1 routes to direct_palm.run_direct
#  - direct_palm survives execution inside a forkserver worker pool process
#  - downstream ESMDA + viz still consume true_state.nc as before

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

PROJECT_STORAGE="/projects/urbanair"
SCRATCH_DIR="/scratch/${USER}"
RUN_TEMP_DIR="${SCRATCH_DIR}/m2_smoke/${SLURM_JOB_ID}"
mkdir -p job_scripts/delftblue/out_files "${PROJECT_STORAGE}" "${RUN_TEMP_DIR}"

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Repo: $(pwd)"
echo "Results: ${PROJECT_STORAGE}  (temp: ${RUN_TEMP_DIR})"
echo "Extra Hydra overrides: $*"

module purge

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

# Direct-run uses /scratch under the slurm job dir (no /tmp rm-trap, so a
# failure leaves the tempdirs around for forensics). M2 is short enough that
# even on local SSD the win over palmrun is dominated by the elimination of
# palmrun's preamble, not the I/O placement.
export PYPALM_FAST_IO_CATALOG="${RUN_TEMP_DIR}/fast_io"
mkdir -p "${PYPALM_FAST_IO_CATALOG}"

# === The bit that activates this milestone. ===
export PYPALM_USE_DIRECT_RUN=1

if [ -d libs/pypalm/palm_model_system ]; then
    find libs/pypalm/palm_model_system -type f -name CMakeCache.txt -delete
fi

pixi run -e delftblue -- bash -c '
  unset OPAL_PREFIX CC CXX F77 F90 FC
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  exec python -u scripts/run_time_varying_parameters_rollout_esmda.py \
      size=tiny \
      domain.nz=16 \
      model@truth_model=pypalm \
      model@assim_model=pypalm \
      ensemble.ensemble_size=2 \
      ensemble.num_parallel_processes=2 \
      esmda.num_assimilation_windows=1 \
      esmda.num_steps=1 \
      localization=correlation \
      paths.base_results_dir='"${PROJECT_STORAGE}"' \
      +truth_model.forward_model.temp_dir='"${RUN_TEMP_DIR}"' \
      +assim_model.forward_model.temp_dir='"${RUN_TEMP_DIR}"' \
      "$@"
' bash "$@"
