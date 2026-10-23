#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_train_local_fno.slurm paths.data_dir=<corpus> prechunk.output_root=<copy>
# Trains Local-FNO with the paper's recipe
# (configs/surrogate/baselines/local_fno/train.yaml) on one A100: its largest
# realistic-corpus batch peaks at ~29 GB, and 12 epochs take ~3 h there.
# Resubmit the same line to resume.
# On the realistic corpus (as the DFT):
#   R=/projects/prjs2304/urbanair
#   sbatch job_scripts/snellius/surrogate_train_local_fno.slurm \
#     paths.weights_dir=$R/model_weights \
#     paths.data_dir=$R/training_data/pyudales_realistic \
#     prechunk.output_root=$R/training_data/pyudales_realistic_frames
#SBATCH --job-name=surrogate_train_local_fno
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# Batches change shape with the grid: expandable segments avoid fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name surrogate/baselines/local_fno/train trainer.checkpoint_every=1 "$@"
