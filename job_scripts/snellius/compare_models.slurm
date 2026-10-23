#!/bin/bash
#   sbatch job_scripts/snellius/compare_models.slurm 'models=[pyudales,pypalm]' replicas=2
#SBATCH --job-name=compare_models
#SBATCH --partition=rome
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run bash workflows/compare_models.sh "$@"
