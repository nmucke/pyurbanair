#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_generate_data.slurm data.sharding.num_shards=4 data.sharding.shard_index=0
#SBATCH --job-name=surrogate_generate_data
#SBATCH --partition=compute-p2
#SBATCH --account=innovation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/generate_data.py "$@"
