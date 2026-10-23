#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_train_ssrolling.slurm <train_roll1|finetune_roll3|train_roll3> paths.data_dir=<corpus> prechunk.output_root=<copy>
# Trains SSRollingUrbanNet (configs/surrogate/baselines/ssrolling/<stage>.yaml)
# on one A100 80 GB (batch 2 on the 2.8 Mcell realistic grids peaks at ~39 GB
# in Roll-1, ~31 GB in the checkpointed Roll-3): the paper runs train_roll1,
# then finetune_roll3 from it; train_roll3 is the from-scratch arm. Resubmit
# the same line to resume. Epoch caps for the realistic corpus: see the
# Snellius script.
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
# Variable grids: fewer fragmentation OOMs; no cap on cuDNN's cached plans.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=0
run python -u scripts/surrogate/train.py --config-name "surrogate/baselines/ssrolling/${stage}" trainer.checkpoint_every=1 "$@"
