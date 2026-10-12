#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_train_dft.slurm paths.data_dir=<corpus> pretrained_dir=<autoencoder> prechunk.output_root=<copy>
# Fine-tunes a pretrained autoencoder into the DFT stepper
# (configs/surrogate/train_dft.yaml) on one A100. Resubmit the same line to
# resume.
#SBATCH --job-name=surrogate_train_dft
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/train_dft trainer.checkpoint_every=1 "$@"
