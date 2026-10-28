#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_evaluate_latent_generator_ensemble.slurm [overrides]
#   sbatch --array=2 job_scripts/delftblue/surrogate_evaluate_latent_generator_ensemble.slurm \
#       'latent_generator_ensemble.samples=[0,1]' latent_generator_ensemble.num_draws=8
# Evaluates the compact latent generators against the seed ensemble
# pyudales_realistic_seed_ensemble with
# scripts/surrogate/evaluate_latent_generator_ensemble.py (configs/surrogate/eval.yaml,
# block `latent_generator_ensemble`), one model per array task on one A100.
# Outputs in <model dir>/eval_seed_ensemble. Then compare them on a CPU node:
#   sbatch --dependency=afterok:<job id> job_scripts/delftblue/surrogate_compare_latent_generator_ensembles.slurm
#SBATCH --job-name=surrogate_evaluate_latent_generator_ensemble
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --array=0-6
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7900M
#SBATCH --time=05:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%A_%a.out

# Array task i -> model i (keep in step with the compare job's list).
models=(
  lg_b_realistic_a_h320_w1024
  lg_b_realistic_b_h256_m512
  best_latent_flow_matching_realistic
  lg_b_realistic_h512_l6_xpred
  lg_b_realistic_h512_l6_ts0p5
  lg_b_realistic_h512_l6_ts3
  latent_generator_b_realistic_adaln_h176
)
model="${models[${SLURM_ARRAY_TASK_ID:-2}]}"

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/evaluate_latent_generator_ensemble.py \
  "latent_generator_ensemble.model_dir=\${paths.results_root}/\${paths.weights_dir}/${model}" "$@"
