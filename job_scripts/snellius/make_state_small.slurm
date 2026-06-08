#!/bin/bash
# Write a reduced copy of the time-varying ground-truth state on Snellius.
#
#     sbatch job_scripts/snellius/make_state_small.slurm
#
# Drives scripts/make_state_small.py, which reads
#   ${PROJECT_STORAGE}/ground_truth/pyudales_time_varying/state.nc  (read-only)
# and writes state_small.nc next to it, keeping only u, v, w and time in
# [0, 1800]. The original state.nc is left untouched.
#SBATCH --job-name=make_state_small
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1750M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-make_state_small-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-make_state_small-%j.err

set -euo pipefail

ENV="snellius"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/snellius/out_files

module purge
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"

pixi run -e "${ENV}" -- python -u scripts/make_state_small.py

echo "Done — $(date)"
