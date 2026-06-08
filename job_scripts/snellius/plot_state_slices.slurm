#!/bin/bash
# Plot horizontal (z-constant) slices of a uDALES ground-truth state.nc and
# animate a single z slice over time. Streams the data frame-by-frame, so it
# stays well under the requested memory regardless of the number of time steps.
#
#     sbatch job_scripts/snellius/plot_state_slices.slurm
#
# Optional positional args (override the CONFIG defaults below):
#     sbatch job_scripts/snellius/plot_state_slices.slurm <state.nc> <var> <z_index>
#SBATCH --job-name=plot_state_slices
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-plot_state_slices-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-plot_state_slices-%j.err

set -euo pipefail

# ============================================================================
# CONFIG
# ============================================================================
PROJECT_STORAGE="/projects/prjs2075/urbanair"
STATE_NC="${1:-${PROJECT_STORAGE}/ground_truth_small/pyudales_time_varying/state_final.nc}"
VAR="${2:-vel_magnitude}"        # vel_magnitude | u | v | w | pres
Z_INDEX="${3:-}"                 # empty -> mid level
OUT_DIR="${4:-$(dirname "${STATE_NC}")/plots}"   # 4th arg overrides output dir
FPS="${5:-15}"                   # mp4 frames/sec; lower = slower playback
# ============================================================================

ENV="snellius"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/snellius/out_files "${OUT_DIR}"

module purge
# ffmpeg for mp4 encoding (not in the pixi env); expose its path to matplotlib.
module load 2023 FFmpeg/6.0-GCCcore-12.3.0
export FFMPEG_BIN="$(command -v ffmpeg)"
echo "ffmpeg: ${FFMPEG_BIN}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "State: ${STATE_NC}"
echo "Var=${VAR}  z_index='${Z_INDEX}'  out=${OUT_DIR}"

pixi run -e "${ENV}" -- python -u scripts/plot_state_slices.py \
    "${STATE_NC}" "${OUT_DIR}" "${VAR}" "${Z_INDEX}" "${FPS}"

echo "Done — plots under ${OUT_DIR}"
