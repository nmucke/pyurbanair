#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_train_latent_generator.slurm [overrides]
# Trains the latent flow-matching generator
# (configs/surrogate/train_latent_generator.yaml) on one A100, from the latent
# cache made by surrogate_prepare_latents.slurm. Resubmit the same line to
# resume.
#SBATCH --job-name=surrogate_train_latent_generator
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name surrogate/train_latent_generator trainer.checkpoint_every=1 "$@"
