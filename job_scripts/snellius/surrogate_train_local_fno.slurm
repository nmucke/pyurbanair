#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_train_local_fno.slurm paths.data_dir=<corpus>
# Trains Local-FNO with the paper's recipe
# (configs/surrogate/baselines/local_fno/train.yaml) on one A100. Resubmit the
# same line to resume.
#SBATCH --job-name=surrogate_train_local_fno
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name surrogate/baselines/local_fno/train trainer.checkpoint_every=1 "$@"
