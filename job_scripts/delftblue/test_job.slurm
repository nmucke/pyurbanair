#!/bin/bash
#SBATCH --job-name=udales_201
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-udales_201-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-udales_201-%j.err

module purge
pixi run -e delftblue -- python scripts/main_udales.py
