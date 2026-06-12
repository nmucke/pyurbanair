#!/bin/bash
# Generate neural-surrogate TRAINING DATA on Snellius. Self-contained submit
# file (like job_scripts/snellius/ground_truth.slurm): edit the CONFIG block
# below, then submit it directly:
#
#     sbatch job_scripts/snellius/generate_training_data.slurm
#
# Extra Hydra overrides may be appended on the command line and take precedence:
#
#     sbatch job_scripts/snellius/generate_training_data.slurm training_data.seed=1
#
# It drives scripts/generate_training_data.py with `training_data=${SIZE}`. That
# overlay samples inflow params (inflow_angle / velocity_magnitude) from the
# UniformExternalAR2 prior, runs ALL train+val+test members in ONE parallel
# ensemble (one pyudales solve per member, ncpu=1), then partitions the
# per-member NetCDFs into state/{split}/ + param/{split}/ under ${OUTPUT_DIR},
# plus per-split figures/animations. The size overlays force model=pyudales.
#
# Memory: the ensemble writes one NetCDF per member to a staging dir and the
# partition step re-opens them one at a time, so peak RAM is ~num_parallel
# member states, not the whole dataset.
#SBATCH --job-name=gen_training_data
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1750M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-gen_training_data-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-gen_training_data-%j.err

set -euo pipefail

# ============================================================================
# CONFIG — edit these for your training-data run.
# ============================================================================
# Dataset size overlay (conf/training_data/<SIZE>.yaml): tiny|small|medium|large|xlarge.
# Each sets num_train/val/test, the grid, and the sampling ranges; all force pyudales.
SIZE="medium"

# Number of ensemble members run concurrently (one pyudales solve per worker at
# ncpu=1). Keep <= --cpus-per-task above. Overrides the size overlay's default.
NUM_PARALLEL=64
# ============================================================================

# Snellius is CPU-only here; pixi env "snellius" sets JAX_PLATFORMS=cpu.
ENV="snellius"
PROJECT_STORAGE="/projects/prjs2075/urbanair"
# Final training data lands here (state/, param/, params.nc, figures, animations).
# Routed to project storage rather than the home quota.
OUTPUT_DIR="${PROJECT_STORAGE}/training_data/pyudales_${SIZE}"

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p job_scripts/snellius/out_files "${OUTPUT_DIR}"

# Heavy per-member solver I/O goes on /scratch-shared, not the home quota /
# project space. Cleaned up on success; left behind on failure for inspection.
RUN_TEMP_DIR="/scratch-shared/${USER}/urbanair_temp/${SLURM_JOB_ID:-$$}"
mkdir -p "${RUN_TEMP_DIR}"
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "SNELLIUS training-data gen: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Size=${SIZE} model=pyudales parallel=${NUM_PARALLEL} cores=${SLURM_CPUS_PER_TASK:-?}"
echo "Output: ${OUTPUT_DIR}  (temp: ${RUN_TEMP_DIR})"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# Keep BLAS single-threaded so the parallel ensemble workers (one solve per core)
# don't oversubscribe via nested BLAS threads; line-buffer Python so a crash
# keeps its traceback.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# hydra.run.dir is forced to an absolute path: Hydra's default run dir is
# ${paths.results_dir} = .temp/<model>, and .temp is a symlink into a per-job
# /scratch-shared dir that the cleanup trap removes on success -- so between runs
# it dangles and Hydra's mkdir under it fails. The other Snellius submit files
# override hydra.run.dir for the same reason.
#
# pyudales solver flags: each member runs serially (ncpu=1) with its scratch +
# fielddump outputs redirected under this job's private /scratch-shared dir (the
# ensemble nests per-member subdirs beneath it).
pixi run -e "${ENV}" -- python -u \
    scripts/generate_training_data.py \
    "training_data=${SIZE}" \
    "training_data.num_parallel_processes=${NUM_PARALLEL}" \
    "training_data.output_dir=${OUTPUT_DIR}" \
    "hydra.run.dir=${OUTPUT_DIR}/hydra" \
    "model.forward_model.ncpu=1" \
    "model.forward_model.temp_dir=${RUN_TEMP_DIR}" \
    "+model.forward_model.output_dir=${RUN_TEMP_DIR}/outputs" \
    "$@"

echo "Done -- training data under ${OUTPUT_DIR}"
