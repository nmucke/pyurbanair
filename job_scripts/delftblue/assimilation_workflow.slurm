#!/bin/bash
#   sbatch job_scripts/delftblue/assimilation_workflow.slurm smoother model@assim_model=pylbm ensemble.ensemble_size=64
#SBATCH --job-name=assimilation_workflow
#SBATCH --partition=compute-p2
#SBATCH --account=innovation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run bash workflows/assimilation_workflow.sh "$@"
