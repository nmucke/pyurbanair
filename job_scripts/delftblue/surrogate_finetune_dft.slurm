#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_finetune_dft.slurm [prechunk.output_root=<copy>] [pretrained_dir=<dft>] [name=<out>]
# Fine-tunes the trained DFT stepper dft_b on pyudales_barcelona_finetune
# (configs/surrogate/finetune_dft.yaml) on one A100 (80 GB). Output in
# /projects/urbanair/model_weights/dft_b_ft_barcelona. Resubmit the same line
# to resume.
#SBATCH --job-name=surrogate_finetune_dft
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/finetune_dft trainer.checkpoint_every=1 \
  paths.weights_dir=/projects/urbanair/model_weights \
  paths.training_data_dir=/projects/urbanair/training_data "$@"
