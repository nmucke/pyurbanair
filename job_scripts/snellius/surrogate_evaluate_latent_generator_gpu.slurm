#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_evaluate_latent_generator_gpu.slurm [latent_generator.model_dir=<model dir>]
# Evaluates the latent generator (default best_latent_flow_matching_realistic) with
# scripts/surrogate/evaluate_latent_generator.py on one A100
# (configs/surrogate/eval.yaml, block `latent_generator`). Outputs in <model dir>/eval.
#SBATCH --job-name=surrogate_evaluate_latent_generator_gpu
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); evaluate in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/evaluate_latent_generator.py \
  'latent_generator.model_dir=${paths.results_root}/${paths.weights_dir}/best_latent_flow_matching_realistic' "$@"
