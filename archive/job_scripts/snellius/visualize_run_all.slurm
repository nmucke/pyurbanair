#!/bin/bash
# SNELLIUS batch visualizer. Runs scripts/figure_creation/visualize_run.py over EVERY run folder
# in a parent directory (default /projects/prjs2075/urbanair/assim_from_ground_truth).
# Each subfolder that holds a run_summary.yaml is treated as one run; figures land
# under result_figures/<run folder name>.
#
# Submit from the repo root:
#
#     sbatch job_scripts/snellius/visualize_run_all.slurm
#     sbatch job_scripts/snellius/visualize_run_all.slurm /some/other/parent --no-truth
#
# First arg (optional) overrides the parent directory; any further args are
# forwarded verbatim to visualize_run.py for every run (e.g. --no-truth, --z-level).
#
# Light job: each run reads only the small param/summary artifacts plus a single
# truth frame, so a handful of cores on rome is plenty even run back-to-back.
#SBATCH --job-name=visualize_run_all
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-visualize_run_all-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-visualize_run_all-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PARENT="${1:-/projects/prjs2075/urbanair/assim_from_ground_truth}"
# Drop the parent arg (if given) so the rest passes through to the Python script.
[ "$#" -ge 1 ] && shift

if [ ! -d "$PARENT" ]; then
    echo "parent directory not found: $PARENT" >&2
    exit 1
fi

mkdir -p job_scripts/snellius/out_files

module purge
# Single-process plotting; keep BLAS single-threaded and line-buffer Python.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "SNELLIUS visualize_run_all: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Parent folder: $PARENT"
[ "$#" -gt 0 ] && echo "Extra args: $*"

n_ok=0
n_skip=0
n_fail=0
for run_dir in "$PARENT"/*/; do
    run_dir="${run_dir%/}"
    if [ ! -f "$run_dir/run_summary.yaml" ]; then
        echo "-- skip (no run_summary.yaml): $run_dir"
        n_skip=$((n_skip + 1))
        continue
    fi
    echo "=== visualizing: $run_dir ==="
    if pixi run -e snellius -- python -u scripts/figure_creation/visualize_run.py "$run_dir" "$@"; then
        n_ok=$((n_ok + 1))
    else
        echo "!! failed: $run_dir" >&2
        n_fail=$((n_fail + 1))
    fi
done

echo "Done -- ok=$n_ok skipped=$n_skip failed=$n_fail. Figures under result_figures/"
