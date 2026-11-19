#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_continue_dft.slurm name=<out> trainer.rollout_loss=<final|sum> [overrides...]
# Continues training a trained DFT stepper (configs/surrogate/continue_dft.yaml,
# default dft_s on pyudales_idealized; pass pretrained_dir/paths.data_dir and the
# batch settings for dft_b) on one A100 (80 GB). Output in
# /projects/urbanair/model_weights/<name>. Resubmit the same line to resume.
#SBATCH --job-name=surrogate_continue_dft
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/continue_dft \
  paths.weights_dir=/projects/urbanair/model_weights \
  paths.training_data_dir=/projects/urbanair/training_data "$@"
