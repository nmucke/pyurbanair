#!/bin/bash
# Trim the spin-up transient from a ground-truth state, then visualize the
# trimmed result. Submit directly:
#
#     sbatch job_scripts/delftblue/trim_and_visualize.slurm
#
# Step 1 runs scripts/trim_spinup.py with --state-only, dropping the leading
# 50 s of spin-up from state_not_spunup.nc and writing state.nc into GT_DIR
# (params.nc is copied through unchanged). Step 2 runs
# scripts/visualize_ground_truth.py on the trimmed GT_DIR.
#SBATCH --job-name=trim_and_visualize
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-trim_and_visualize-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-trim_and_visualize-%j.err

set -euo pipefail

# ============================================================================
# CONFIG
# ============================================================================
ENV="delftblue"
GT_DIR="/projects/urbanair/ground_truth"
SPINUP_TIME=50.0
# ============================================================================

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/delftblue/out_files

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "GT_DIR=${GT_DIR}  spinup_time=${SPINUP_TIME}"

module purge
export PYTHONUNBUFFERED=1

# Step 1: trim the spin-up from the state only (params copied unchanged).
pixi run -e "${ENV}" -- python -u scripts/trim_spinup.py \
    --state "${GT_DIR}/state_not_spunup.nc" \
    --params "${GT_DIR}/params.nc" \
    --spinup-time "${SPINUP_TIME}" \
    --state-only \
    --output-dir "${GT_DIR}"

# Step 2: visualize the trimmed ground truth.
pixi run -e "${ENV}" -- python -u scripts/visualize_ground_truth.py \
    "${GT_DIR}"

echo "Done — trimmed state.nc and figures under ${GT_DIR}"
