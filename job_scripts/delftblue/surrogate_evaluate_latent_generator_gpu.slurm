#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_evaluate_latent_generator_gpu.slurm [latent_generator.model_dir=<model dir>]
# Evaluates the latent generator (default best_latent_flow_matching_realistic) with
# scripts/surrogate/evaluate_latent_generator.py on one A100
# (configs/surrogate/eval.yaml, block `latent_generator`). Outputs in <model dir>/eval.
#SBATCH --job-name=surrogate_evaluate_latent_generator_gpu
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7900M
#SBATCH --time=08:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/evaluate_latent_generator.py \
  'latent_generator.model_dir=${paths.results_root}/${paths.weights_dir}/best_latent_flow_matching_realistic' "$@"
