#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_prepare_latents.slurm latent_cache.output_root=<cache>
# Encodes every train/val state with the frozen autoencoder once, for
# train_latent_generator with the same latent_cache.output_root (and the same
# autoencoder_dir / paths.data_dir overrides). Resubmit the same line to resume.
#SBATCH --job-name=surrogate_prepare_latents
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); encode in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name surrogate/train_latent_generator "$@" latent_cache.prepare_only=true
