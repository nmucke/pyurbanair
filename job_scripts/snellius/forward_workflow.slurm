#!/bin/bash
#   sbatch job_scripts/snellius/forward_workflow.slurm model=pylbm params=dynamic_truth
#SBATCH --job-name=forward_workflow
#SBATCH --partition=rome
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run bash workflows/forward_workflow.sh "$@"
