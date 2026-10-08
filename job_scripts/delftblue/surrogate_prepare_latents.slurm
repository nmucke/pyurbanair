#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_prepare_latents.slurm latent_cache.output_root=<cache>
# Encodes every train/val state with the frozen autoencoder once, for
# train_latent_generator with the same latent_cache.output_root (and the same
# autoencoder_dir / paths.data_dir overrides). Resubmit the same line to resume.
#SBATCH --job-name=surrogate_prepare_latents
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/train_latent_generator "$@" latent_cache.prepare_only=true
