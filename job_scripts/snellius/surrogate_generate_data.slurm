#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_generate_data.slurm data.sharding.num_shards=4 data.sharding.shard_index=0
#SBATCH --job-name=surrogate_generate_data
#SBATCH --partition=rome
#SBATCH --account=tdsr72361
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
run python -u scripts/surrogate/generate_data.py "$@"
