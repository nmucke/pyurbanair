#!/bin/bash
#SBATCH --job-name=esmda_udales
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-esmda_udales-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-esmda_udales-%j.err

module purge
pixi run -e delftblue -- python scripts/esmda_udales.py
