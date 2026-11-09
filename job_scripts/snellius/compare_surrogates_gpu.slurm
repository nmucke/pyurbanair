#!/bin/bash
#   sbatch job_scripts/snellius/compare_surrogates_gpu.slurm 'models=[...]' [overrides...]
# compare_surrogates.slurm on one A100 in the CUDA env, for device=cuda (the
# snellius env has CPU-only torch; `pixi install -e cuda` once on a login node).
#SBATCH --job-name=compare_surrogates_gpu
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu bash workflows/compare_surrogates.sh "$@"
