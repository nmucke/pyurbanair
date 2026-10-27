#!/bin/bash
# Seed ensemble of a corpus's test split (configs/surrogate/generate_seed_ensemble.yaml):
# array task i runs every N-th (sample, member) pair, N = the array size; a
# rerun skips members on disk. 640 runs (10 x 64) of ~30-45 min on 16 cores.
#   sbatch job_scripts/snellius/surrogate_generate_seed_ensemble.slurm
#   sbatch --array=0-15 job_scripts/snellius/surrogate_generate_seed_ensemble.slurm \
#       'data.seed_ensemble.samples=[6]' 'data.seed_ensemble.vary=[initial_condition]' \
#       'paths.results_dir=${data.seed_ensemble.reference_dir}_seed_ensemble_ic'
#SBATCH --job-name=surrogate_generate_seed_ensemble
#SBATCH --partition=rome
#SBATCH --account=tdsei19851
#SBATCH --array=0-63
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=12:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%A_%a.out

source job_scripts/snellius/env.sh
run python -u scripts/surrogate/generate_data.py --config-name surrogate/generate_seed_ensemble \
    data.sharding.num_shards="${SLURM_ARRAY_TASK_COUNT}" \
    data.sharding.shard_index="${SLURM_ARRAY_TASK_ID}" "$@"
