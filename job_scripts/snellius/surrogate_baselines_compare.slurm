#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_baselines_compare.slurm 'models=[...]'
#SBATCH --job-name=surrogate_baselines_compare
#SBATCH --partition=rome
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=08:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run python -u scripts/surrogate/baselines/compare.py "$@"
