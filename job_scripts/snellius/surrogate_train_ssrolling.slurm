#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_train_ssrolling.slurm <train_roll1|finetune_roll3|train_roll3> paths.data_dir=<corpus>
# Trains SSRollingUrbanNet (configs/surrogate/baselines/ssrolling/<stage>.yaml)
# on one A100: the paper runs train_roll1, then finetune_roll3 from it;
# train_roll3 is the from-scratch arm. Resubmit the same line to resume.
#SBATCH --job-name=surrogate_train_ssrolling
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

stage=${1:-}
if [[ ! "${stage}" =~ ^(train_roll1|finetune_roll3|train_roll3)$ ]]; then
    echo "usage: sbatch $0 <train_roll1|finetune_roll3|train_roll3> [overrides...]" >&2
    exit 1
fi
shift

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name "surrogate/baselines/ssrolling/${stage}" trainer.checkpoint_every=1 "$@"
