#!/bin/bash
# Plot horizontal (z-constant) slices of a uDALES ground-truth state.nc and
# animate a single z slice over time. Streams the data frame-by-frame, so it
# stays well under the requested memory regardless of the number of time steps.
#
#     sbatch job_scripts/delftblue/plot_state_slices.slurm
#
# Optional positional args (override the CONFIG defaults below):
#     sbatch job_scripts/delftblue/plot_state_slices.slurm <state.nc> <var> <z_index>
#
# NB: DelftBlue has NO ffmpeg module (checked across the 2024r1/2025 stacks)
# and the pixi env does not ship one, so the mp4 animation step needs an ffmpeg
# binary from elsewhere: either `pixi add ffmpeg` (conda-forge) once on a login
# node, or export FFMPEG_BIN=<path> at submit time. Without one the static
# slice plots are still written; the animation step then fails.
#SBATCH --job-name=plot_state_slices
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=3900M
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-plot_state_slices-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-plot_state_slices-%j.err

set -euo pipefail

# ============================================================================
# CONFIG
# ============================================================================
PROJECT_STORAGE="/projects/urbanair"
STATE_NC="${1:-${PROJECT_STORAGE}/ground_truth/state.nc}"
VAR="${2:-vel_magnitude}"        # vel_magnitude | u | v | w | pres
Z_INDEX="${3:-}"                 # empty -> mid level
OUT_DIR="${4:-$(dirname "${STATE_NC}")/plots}"   # 4th arg overrides output dir
FPS="${5:-15}"                   # mp4 frames/sec; lower = slower playback
# ============================================================================

ENV="delftblue"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p job_scripts/delftblue/out_files "${OUT_DIR}"

module purge

# Resolve an ffmpeg for the mp4 encoding (no system module on DelftBlue):
# FFMPEG_BIN if exported, else the pixi env's binary, else anything on PATH.
REPO_FFMPEG="$(pwd)/.pixi/envs/${ENV}/bin/ffmpeg"
if [ -z "${FFMPEG_BIN:-}" ]; then
  if [ -x "${REPO_FFMPEG}" ]; then
    FFMPEG_BIN="${REPO_FFMPEG}"
  else
    FFMPEG_BIN="$(command -v ffmpeg || true)"
  fi
fi
if [ -n "${FFMPEG_BIN}" ]; then
  export FFMPEG_BIN
  echo "ffmpeg: ${FFMPEG_BIN}"
else
  echo "warning: no ffmpeg found (no module on DelftBlue) -- the animation step" >&2
  echo "         will fail after the static plots; run \`pixi add ffmpeg\` on a" >&2
  echo "         login node or export FFMPEG_BIN=<path> to fix this." >&2
fi

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
