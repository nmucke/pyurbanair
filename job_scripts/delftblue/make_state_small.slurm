#!/bin/bash
# Write a reduced copy of the time-varying ground-truth state on DelftBlue.
#
#     sbatch job_scripts/delftblue/make_state_small.slurm
#
# Drives scripts/make_state_small.py, which streams a reduced copy of a large
# ground-truth state (keeping only u, v, w and a clipped time range). The
# source/destination paths and the time range are hardcoded constants at the
# top of the script (currently ground_truth/pyudales_time_varying/state_full.nc
# -> state.nc, t in [0, 1000]) — edit them there before submitting. The source
# file is opened read-only and left untouched.
#SBATCH --job-name=make_state_small
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-make_state_small-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-make_state_small-%j.err

set -euo pipefail

ENV="delftblue"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/delftblue/out_files

module purge
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"

pixi run -e "${ENV}" -- python -u scripts/make_state_small.py

echo "Done — $(date)"
