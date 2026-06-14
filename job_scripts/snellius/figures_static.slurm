#!/bin/bash
# SNELLIUS: generate ALL static talk figures (Blocks A/B/C + summary + tables +
# NOTES) for the EnKF-2026 deck, per docs/figure_specs.md, into figures/.
#
# Runs scripts/make_all_figures.py --heavy, which loads the posterior-mean
# rollouts (~0.25-2.6 GB each) and the wide ground truth (~0.9 GB), interpolates
# everything onto the common truth grid and computes the field metrics. Single
# process; the cores mainly buy RAM. No ffmpeg needed (animations are separate).
#
# Submit from the repo root:
#     sbatch job_scripts/snellius/figures_static.slurm
#     sbatch job_scripts/snellius/figures_static.slurm --out figures --no-mask
#
#SBATCH --job-name=figures_static
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # rome billing minimum; peak RAM here is only a few GB
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=1792M         # 16 x 1792M = ~28 GB, billed as 16 cores (the default ratio)
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-figures_static-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-figures_static-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p job_scripts/snellius/out_files

module purge
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "SNELLIUS figures_static: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
[ "$#" -gt 0 ] && echo "Args: $*"

pixi run -e snellius -- python -u scripts/make_all_figures.py --heavy "$@"

echo "Done -- static figures under figures/  ($(date))"
