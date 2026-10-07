#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_evaluate_stepper.slurm
#SBATCH --job-name=surrogate_evaluate_stepper
#SBATCH --partition=compute-p1
#SBATCH --account=innovation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=08:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/evaluate_stepper.py "$@"
