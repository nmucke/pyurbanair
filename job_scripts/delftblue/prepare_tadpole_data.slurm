#!/bin/bash
# Optional CPU-only preparation; use the same chunk overrides for training.
#   sbatch job_scripts/delftblue/prepare_tadpole_data.slurm
# Then: PRECHUNK_DATA=true sbatch job_scripts/delftblue/pretrain_tadpole_ae.slurm train
#SBATCH --job-name=tadpole_data
#SBATCH --partition=compute-p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=2-00:00:00
#SBATCH --account=research-ceg-gse
#SBATCH --output=slurm-tadpole-data-%j.out
#SBATCH --error=slurm-tadpole-data-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repo root}"
DATA_ROOT="${DATA_ROOT:-/projects/urbanair/training_data/pyudales_realistic}"
PRECHUNK_ROOT="${PRECHUNK_ROOT:-/scratch/${USER}/training_data/$(basename "${DATA_ROOT}")_rechunked}"
PIXI_ENV="${PIXI_ENV:-delftblue}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun pixi run --as-is -e "${PIXI_ENV}" python -u \
    scripts/neural_surrogate/pretrain_autoencoder.py \
    "dataset.root_dir=${DATA_ROOT}" \
    "prechunk.output_root=${PRECHUNK_ROOT}" \
    "hydra.run.dir=${SLURM_SUBMIT_DIR}/.temp/tadpole-preparation-${SLURM_JOB_ID}" \
    "$@" prechunk.enabled=true prechunk.prepare_only=true
