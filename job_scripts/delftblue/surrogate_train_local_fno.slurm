#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_train_local_fno.slurm paths.data_dir=<corpus> prechunk.output_root=<copy>
# Trains Local-FNO with the paper's recipe
# (configs/surrogate/baselines/local_fno/train.yaml) on one A100: its largest
# realistic-corpus batch peaks at ~29 GB. Resubmit the same line to resume.
#SBATCH --job-name=surrogate_train_local_fno
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
# Batches change shape with the grid: expandable segments avoid fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run python -u scripts/surrogate/train.py --config-name surrogate/baselines/local_fno/train trainer.checkpoint_every=1 "$@"
