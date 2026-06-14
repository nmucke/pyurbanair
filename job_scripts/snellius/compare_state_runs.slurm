#!/bin/bash
# SNELLIUS comparison of multiple STATE-estimation ESMDA runs, via
# scripts/compare_state_runs.py. Scans a directory of run folders, reads each
# run's run_summary.yaml and draws grouped metric-comparison figures + a summary
# table / CSV under result_figures/comparison_<mode> (by default).
#
# All arguments pass straight through to the Python script. The most useful is
# --mode {ic,all,both}: ic = initial-condition runs only, all = state-throughout
# runs only, both = every run together (default). Submit from the repo root:
#
#     sbatch job_scripts/snellius/compare_state_runs.slurm                 # both, default root
#     sbatch job_scripts/snellius/compare_state_runs.slurm --mode ic
#     sbatch job_scripts/snellius/compare_state_runs.slurm --mode all
#     sbatch job_scripts/snellius/compare_state_runs.slurm --root /path/to/runs --mode both
#
# Very light: only reads the small run_summary.yaml of each run (no NetCDF /
# ensemble reads), so one core and a few minutes is ample. Incomplete runs
# (no metrics yet -- e.g. mid-transfer) are skipped automatically.
#SBATCH --job-name=compare_state_runs
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-compare_state_runs-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-compare_state_runs-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p job_scripts/snellius/out_files

module purge
# Single-process plotting; keep BLAS single-threaded and line-buffer Python.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "SNELLIUS compare_state_runs: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
[ "$#" -gt 0 ] && echo "Args: $*"

# All arguments forward straight through (e.g. --mode, --root, --out).
pixi run -e snellius -- python -u scripts/compare_state_runs.py "$@"

echo "Done -- comparison under result_figures/"
