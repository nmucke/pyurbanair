#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_train.slurm --config-name surrogate/train_autoencoder
#SBATCH --job-name=surrogate_train
#SBATCH --partition=rome
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run python -u scripts/surrogate/train.py "$@"
