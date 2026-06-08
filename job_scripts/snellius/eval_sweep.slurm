#!/bin/bash
# Post-process the rollout-ESMDA sweep on Snellius: compute every metric + metric
# time series, then draw all comparison figures. Two stages of the three-script
# pipeline (run_esmda.py is the first, submitted by the sweep launchers):
#
#     sbatch job_scripts/snellius/eval_sweep.slurm
#
# Extra args are forwarded to BOTH scripts where applicable, e.g. restrict models:
#
#     sbatch job_scripts/snellius/eval_sweep.slurm --models pyudales pylbm
#
# Stage 1 -- scripts/compute_sweep_metrics.py: reads the (large) posterior states
#   under the project results root and the ground truth, and writes SMALL metric
#   artifacts to pyurbanair/sweep_metrics/. It opens one window's full-ensemble
#   state at a time, so a high-resolution / large-ensemble sweep needs several GB
#   of RAM -- hence this runs on a compute node, NOT the login node.
# Stage 2 -- scripts/compare_sweep_results.py: reads pyurbanair/sweep_metrics/ and
#   writes figures + the big CSV to pyurbanair/comparison/ (light).
#SBATCH --job-name=eval_sweep
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-eval_sweep-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-eval_sweep-%j.err

set -euo pipefail

ENV="snellius"
# Project results root (where run_esmda.py wrote the per-run dirs). Override on
# the command line by exporting RESULTS_ROOT before sbatch if needed.
RESULTS_ROOT="${RESULTS_ROOT:-/projects/prjs2075/urbanair/assim_from_ground_truth}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/snellius/out_files

module purge
export PYTHONUNBUFFERED=1
# Single-process metric computation / plotting; keep BLAS single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} on $(hostname) -- $(date)"
echo "Results root: ${RESULTS_ROOT}"
[ "$#" -gt 0 ] && echo "Extra args: $*"

echo "=== Stage 1: compute_sweep_metrics -> pyurbanair/sweep_metrics/ ==="
pixi run -e "${ENV}" -- python -u scripts/compute_sweep_metrics.py \
    --root "${RESULTS_ROOT}" "$@"

echo "=== Stage 2: compare_sweep_results -> pyurbanair/comparison/ ==="
pixi run -e "${ENV}" -- python -u scripts/compare_sweep_results.py "$@"

echo "Done -- metrics in sweep_metrics/, figures + CSV in comparison/  ($(date))"
