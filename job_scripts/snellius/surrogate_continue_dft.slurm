#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_continue_dft.slurm name=<out> trainer.rollout_loss=<final|sum> [overrides...]
# Continues training a trained DFT stepper (configs/surrogate/continue_dft.yaml,
# default dft_s on pyudales_idealized; pass pretrained_dir/paths.data_dir and the
# batch settings for dft_b) on one H100 (94 GB, as dft_b). Output in
# /projects/prjs2304/urbanair/model_weights/<name>. Resubmit the same line to resume.
#SBATCH --job-name=surrogate_continue_dft
#SBATCH --partition=gpu_h100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name surrogate/continue_dft \
  paths.weights_dir=/projects/prjs2304/urbanair/model_weights \
  paths.training_data_dir=/projects/prjs2304/urbanair/training_data "$@"
