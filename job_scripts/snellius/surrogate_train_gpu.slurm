#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_train_gpu.slurm --config-name surrogate/<config> [overrides...]
# surrogate_train.slurm on one H100 in the CUDA env (the snellius env has
# CPU-only torch; `pixi install -e cuda` once on a login node), e.g. the
# autoencoder. Resubmit the same line to resume.
#SBATCH --job-name=surrogate_train_gpu
#SBATCH --partition=gpu_h100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py "$@"
