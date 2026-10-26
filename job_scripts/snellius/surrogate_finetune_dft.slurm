#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_finetune_dft.slurm [prechunk.output_root=<copy>] [pretrained_dir=<dft>] [name=<out>]
# Fine-tunes the trained DFT stepper dft_b on pyudales_barcelona_finetune
# (configs/surrogate/finetune_dft.yaml) on one A100 (40 GB: one sample per
# micro-batch, the config's default). Output in
# /projects/prjs2304/urbanair/model_weights/dft_b_ft_barcelona. Resubmit the
# same line to resume.
#SBATCH --job-name=surrogate_finetune_dft
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name surrogate/finetune_dft trainer.checkpoint_every=1 \
  paths.weights_dir=/projects/prjs2304/urbanair/model_weights \
  paths.training_data_dir=/projects/prjs2304/urbanair/training_data "$@"
