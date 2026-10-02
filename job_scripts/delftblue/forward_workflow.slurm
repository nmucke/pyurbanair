#!/bin/bash
#   sbatch job_scripts/delftblue/forward_workflow.slurm model=pylbm params=dynamic_truth
#SBATCH --job-name=forward_workflow
#SBATCH --partition=compute-p1
#SBATCH --account=innovation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run bash workflows/forward_workflow.sh "$@"
