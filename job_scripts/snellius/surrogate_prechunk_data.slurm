#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_prechunk_data.slurm paths.data_dir=<dataset> prechunk.output_root=<copy>
# For the DFT's whole-frame copy, start the arguments with
# --config-name surrogate/train_dft (it overrides the one below).
# The copy goes file by file on one core (16 is rome's smallest share).
# Resubmit the same line to resume.
#SBATCH --job-name=surrogate_prechunk_data
#SBATCH --partition=rome
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/train_autoencoder "$@" prechunk.prepare_only=true
