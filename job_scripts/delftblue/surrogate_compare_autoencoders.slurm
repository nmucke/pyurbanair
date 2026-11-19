#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_compare_autoencoders.slurm 'autoencoder_compare.models=[<dir>,<dir>]' [overrides...]
# Compares trained autoencoders on the same snapshots with
# scripts/surrogate/compare_autoencoders.py on one A100
# (configs/surrogate/eval.yaml, block `autoencoder_compare`).
#SBATCH --job-name=surrogate_compare_autoencoders
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7900M
#SBATCH --time=04:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/compare_autoencoders.py "$@"
