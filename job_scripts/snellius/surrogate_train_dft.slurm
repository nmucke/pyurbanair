#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_train_dft.slurm paths.data_dir=<corpus> pretrained_dir=<autoencoder> prechunk.output_root=<copy>
# Fine-tunes a pretrained autoencoder into the DFT stepper
# (configs/surrogate/train_dft.yaml) on one A100. Resubmit the same line to
# resume.
#SBATCH --job-name=surrogate_train_dft
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
run_gpu python -u scripts/surrogate/train.py --config-name surrogate/train_dft trainer.checkpoint_every=1 "$@"
