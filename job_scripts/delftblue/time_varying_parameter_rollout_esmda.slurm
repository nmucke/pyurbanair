#!/bin/bash
#SBATCH --job-name=esmda_udales
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
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

# Each ensemble worker is a separate forked Python process pinned to one
# core. In-rank thread libraries must stay at 1 so 48 workers × 48 BLAS
# threads = 2304 threads don't fight for cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

pixi run -e delftblue -- python scripts/run_time_varying_parameters_rollout_esmda.py \
    --truth-model pyudales --assim-model pyudales \
    --output-dir /projects/urbanair/time_varying_rollout_esmda_pyudales_pyudales
