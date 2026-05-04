#!/bin/bash
#SBATCH --job-name=esmda_udales
#SBATCH --partition=compute
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-esmda_udales-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-esmda_udales-%j.err

set -euo pipefail

# Run from the repo root regardless of where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

mkdir -p job_scripts/delftblue/out_files

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Repo: $(pwd)"

module purge

# Belt-and-suspenders: even though delftblue_activation.sh sets these, set
# them again here so a stale pixi cache or an unsourced activation can't
# bring UCX/InfiniBand back into the MPI stack and crash MPI_Finalize.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# Cap thread fan-out so JAX/BLAS/OpenMP don't oversubscribe the 4 cores
# allocated by --cpus-per-task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

pixi run -e delftblue -- python scripts/run_time_varying_parameters_rollout_esmda.py \
    --truth-model pyudales --assim-model pyudales
