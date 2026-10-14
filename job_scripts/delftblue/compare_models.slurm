#!/bin/bash
#   sbatch job_scripts/delftblue/compare_models.slurm 'models=[pyudales,pypalm]' replicas=2
#SBATCH --job-name=compare_models
#SBATCH --partition=compute-p1
#SBATCH --account=innovation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run bash workflows/compare_models.sh "$@"
