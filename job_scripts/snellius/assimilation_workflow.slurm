#!/bin/bash
#   sbatch job_scripts/snellius/assimilation_workflow.slurm smoother model@assim_model=pylbm ensemble.ensemble_size=64
#SBATCH --job-name=assimilation_workflow
#SBATCH --partition=rome
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run bash workflows/assimilation_workflow.sh "$@"
