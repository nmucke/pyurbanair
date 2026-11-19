#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_compare_autoencoders.slurm 'autoencoder_compare.models=[<dir>,<dir>]' [overrides...]
# Compares trained autoencoders on the same snapshots with
# scripts/surrogate/compare_autoencoders.py on one A100
# (configs/surrogate/eval.yaml, block `autoencoder_compare`).
#SBATCH --job-name=surrogate_compare_autoencoders
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); evaluate in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/compare_autoencoders.py "$@"
