#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_train_latent_generator.slurm [overrides]
# Trains the latent flow-matching generator
# (configs/surrogate/train_latent_generator.yaml) on one A100, from the latent
# cache made by surrogate_prepare_latents.slurm. Resubmit the same line to
# resume.
#SBATCH --job-name=surrogate_train_latent_generator
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/train_latent_generator trainer.checkpoint_every=1 "$@"
