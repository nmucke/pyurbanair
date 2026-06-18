#!/bin/bash
# SNELLIUS: render the talk ANIMATIONS (mp4) for the EnKF-2026 deck via
# scripts/make_animations.py, into figures/{A_param_only,B_state_estimation,
# C_full_joint}/. These are the truth|estimate|error triptychs (Block A & C) and
# the side-by-side methods panel (Block B).
#
# ffmpeg is NOT in the pixi env -- it comes from a system module loaded here and
# exposed as FFMPEG_BIN, which make_animations.py points matplotlib's
# FFMpegWriter at. Loads the truth + posterior-mean rollouts and renders frames,
# so give it memory and time.
#
# Submit from the repo root:
#     sbatch job_scripts/snellius/figures_animations.slurm
#     sbatch job_scripts/snellius/figures_animations.slurm --which A --fps 10
#     sbatch job_scripts/snellius/figures_animations.slurm --frames        # also dump PNGs
#
#SBATCH --job-name=figures_anim
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # rome billing minimum
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=1792M         # 16 x 1792M = ~28 GB, billed as 16 cores
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-figures_anim-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-figures_anim-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p job_scripts/snellius/out_files

module purge
# ffmpeg for matplotlib mp4 writing (NOT in the pixi env).
module load 2023 FFmpeg/6.0-GCCcore-12.3.0
export FFMPEG_BIN="$(command -v ffmpeg)"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "SNELLIUS figures_anim: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "ffmpeg: ${FFMPEG_BIN:-<none>}"
[ "$#" -gt 0 ] && echo "Args: $*"

pixi run -e snellius -- python -u scripts/make_animations.py "$@"

echo "Done -- animations under figures/  ($(date))"
