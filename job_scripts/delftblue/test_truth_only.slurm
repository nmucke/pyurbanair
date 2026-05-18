#!/bin/bash
#SBATCH --job-name=test_truth_udales
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-test_truth-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-test_truth-%j.err

set -euo pipefail

# Run from the repo root regardless of where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

mkdir -p job_scripts/delftblue/out_files

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Repo: $(pwd)"

module purge

# Match the OpenMPI MCA settings used in the full ESMDA job so the MPI
# stack here is identical — we want the test to reproduce (or eliminate)
# the cleanup-time segfault, not paper over it with different settings.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export PYTHONUNBUFFERED=1

# Reproduce window 0 of the rollout truth_model: time-varying params
# drawn from EXTERNAL_PRIORS via truth_ts_model, state=None, sim_time +
# spinup_time runtime. Same code path that segfaulted in jobs 9853728,
# 9857112, 9858376. ~5 minutes if kappa + reverted nudging fixed it.
pixi run -e delftblue -- python -u scripts/test_truth_only.py --model pyudales
