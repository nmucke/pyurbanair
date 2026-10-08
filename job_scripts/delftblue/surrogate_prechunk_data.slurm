#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_prechunk_data.slurm paths.data_dir=<dataset> prechunk.output_root=<copy>
# For the DFT's whole-frame copy, start the arguments with
# --config-name surrogate/train_dft (it overrides the one below).
# One core: the copy goes file by file. Resubmit the same line to resume.
#SBATCH --job-name=surrogate_prechunk_data
#SBATCH --partition=compute-p2
#SBATCH --account=innovation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/train_autoencoder "$@" prechunk.prepare_only=true
