#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_train_ssrolling.slurm <train_roll1|finetune_roll3|train_roll3> paths.data_dir=<corpus>
# Trains SSRollingUrbanNet (configs/surrogate/baselines/ssrolling/<stage>.yaml)
# on one A100: the paper runs train_roll1, then finetune_roll3 from it;
# train_roll3 is the from-scratch arm. Resubmit the same line to resume.
#SBATCH --job-name=surrogate_train_ssrolling
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

stage=${1:-}
if [[ ! "${stage}" =~ ^(train_roll1|finetune_roll3|train_roll3)$ ]]; then
    echo "usage: sbatch $0 <train_roll1|finetune_roll3|train_roll3> [overrides...]" >&2
    exit 1
fi
shift

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name "surrogate/baselines/ssrolling/${stage}" trainer.checkpoint_every=1 "$@"
