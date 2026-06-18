#!/bin/bash
# SNELLIUS single-run visualizer for STATE-estimation ESMDA runs. Regenerates a
# consolidated figure set for ONE scripts/run_esmda.py output folder that
# estimates the state as well as parameters, via scripts/figure_creation/visualize_state_run.py.
# Figures land under result_figures/<case> (case = run folder name by default).
#
# Pass the run folder as the first argument; any further args are forwarded to
# the Python script (e.g. --case, --z-level, --window, --no-truth,
# --no-window-state). Submit from the repo root:
#
#     sbatch job_scripts/snellius/visualize_state_run.slurm /projects/prjs2075/urbanair/assim_with_state/<run>
#     sbatch job_scripts/snellius/visualize_state_run.slurm /projects/.../<run> --window 4 --z-level 2
#
# Mostly light: the parameter / posterior-mean figures read small artifacts. The
# new per-window state figures read the large ensemble-state files, but only a
# single z-slice/time hyperslab at a time (~0.25 GB RAM, ~1 min observed on the
# ens128 runs), so a handful of cores on rome is plenty. Pass --no-window-state
# to skip those entirely.
#SBATCH --job-name=visualize_state_run
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-visualize_state_run-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-visualize_state_run-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch job_scripts/snellius/visualize_state_run.slurm <run_dir> [extra args]" >&2
    exit 1
fi

mkdir -p job_scripts/snellius/out_files

module purge
# Single-process plotting; keep BLAS single-threaded and line-buffer Python.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "SNELLIUS visualize_state_run: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Run folder: $1"
[ "$#" -gt 1 ] && echo "Extra args: ${*:2}"

# First arg is the run folder; the rest pass straight through to the script.
pixi run -e snellius -- python -u scripts/figure_creation/visualize_state_run.py "$@"

echo "Done -- figures under result_figures/"
