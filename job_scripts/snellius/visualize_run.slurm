#!/bin/bash
# SNELLIUS single-run visualizer. Regenerates a consolidated figure set for ONE
# successful scripts/run_esmda.py output folder via scripts/visualize_run.py.
# Figures land under result_figures/<case> (case = run folder name by default).
#
# Pass the run folder as the first argument; any further args are forwarded to
# the Python script (e.g. --case, --no-truth, --z-level). Submit from the repo
# root:
#
#     sbatch job_scripts/snellius/visualize_run.slurm /projects/prjs2075/urbanair/assim_from_ground_truth/<run>
#     sbatch job_scripts/snellius/visualize_run.slurm /projects/.../<run> --case my_case --no-truth
#
# Light job: reads the small param/summary artifacts, the posterior mean/std
# rollout, and (best-effort) a single final frame of the ground truth -- a few
# GB of RAM at most, so a handful of cores on rome is plenty.
#SBATCH --job-name=visualize_run
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-visualize_run-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-visualize_run-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch job_scripts/snellius/visualize_run.slurm <run_dir> [extra args]" >&2
    exit 1
fi

mkdir -p job_scripts/snellius/out_files

module purge
# Single-process plotting; keep BLAS single-threaded and line-buffer Python.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "SNELLIUS visualize_run: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Run folder: $1"
[ "$#" -gt 1 ] && echo "Extra args: ${*:2}"

# First arg is the run folder; the rest pass straight through to the script.
pixi run -e snellius -- python -u scripts/visualize_run.py "$@"

echo "Done -- figures under result_figures/"
