#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_train_ssrolling.slurm <train_roll1|finetune_roll3|train_roll3> paths.data_dir=<corpus> [overrides...]
# Trains SSRollingUrbanNet (configs/surrogate/baselines/ssrolling/<stage>.yaml)
# on one H100: the paper runs train_roll1, then finetune_roll3 from it;
# train_roll3 is the from-scratch arm. Resubmit the same line to resume (from
# the last finished epoch; copy checkpoint.pt aside between jobs).
#
# Realistic corpus (the DFT's data, splits, state_vars and param_vars), with
# the epoch caps of docs/neural_surrogate_baselines.md §3 (the paper's 1000
# epochs / patience 50 would take months here):
#   R=/projects/prjs2304/urbanair
#   D="paths.weights_dir=$R/model_weights paths.data_dir=$R/training_data/pyudales_realistic"
#   D="$D prechunk.output_root=$R/training_data/pyudales_realistic_frames"
#   sbatch job_scripts/snellius/surrogate_train_ssrolling.slurm train_roll1 $D \
#       name=ssrolling_roll1_realistic trainer.num_epochs=15 trainer.patience=5
#   # then, from its weights:
#   sbatch job_scripts/snellius/surrogate_train_ssrolling.slurm finetune_roll3 $D \
#       name=ssrolling_tl_roll3_realistic pretrained_dir=$R/model_weights/ssrolling_roll1_realistic \
#       trainer.num_epochs=7 trainer.patience=5
# ~1.7 h per Roll-1 epoch, ~4.7 h per Roll-3 epoch on an H100: ~26 h + ~33 h.
# Batch 2 on the 2.8 Mcell grids needs the H100 (Roll-1 peaks at ~39 GB);
# Roll-3 activation-checkpoints Aurora (~31 GB, set in its config).
#SBATCH --job-name=surrogate_train_ssrolling
#SBATCH --partition=gpu_h100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

stage=${1:-}
if [[ ! "${stage}" =~ ^(train_roll1|finetune_roll3|train_roll3)$ ]]; then
    echo "usage: sbatch $0 <train_roll1|finetune_roll3|train_roll3> [overrides...]" >&2
    exit 1
fi
shift

source job_scripts/snellius/env.sh
# Variable grids: fewer fragmentation OOMs; no cap on cuDNN's cached plans.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=0
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name "surrogate/baselines/ssrolling/${stage}" trainer.checkpoint_every=1 "$@"
