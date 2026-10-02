#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_train.slurm --config-name surrogate/train_autoencoder
#SBATCH --job-name=surrogate_train
#SBATCH --partition=compute-p2
#SBATCH --account=innovation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py "$@"
