#!/bin/bash
#   sbatch [--dependency=afterok:<evaluation job id>] job_scripts/delftblue/surrogate_compare_latent_generator_ensembles.slurm [overrides]
# Compares the latent generators evaluated by
# surrogate_evaluate_latent_generator_ensemble.slurm
# (scripts/surrogate/compare_latent_generator_ensembles.py, block
# `latent_generator_ensemble_compare` of configs/surrogate/eval.yaml); paired
# differences against R3. CPU only. Outputs in
# <weights dir>/latent_generator_ensemble_comparison.
#SBATCH --job-name=surrogate_compare_latent_generator_ensembles
#SBATCH --partition=compute-p2
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=00:30:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

w='${paths.results_root}/${paths.weights_dir}'
source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/compare_latent_generator_ensembles.py \
  "latent_generator_ensemble_compare.eval_dirs=[$w/lg_b_realistic_a_h320_w1024,$w/lg_b_realistic_b_h256_m512,$w/best_latent_flow_matching_realistic,$w/lg_b_realistic_h512_l6_xpred,$w/lg_b_realistic_h512_l6_ts0p5,$w/lg_b_realistic_h512_l6_ts3,$w/latent_generator_b_realistic_adaln_h176]" \
  'latent_generator_ensemble_compare.tags=[R1,R2,R3,R4,R5a,R5b,h176]' \
  latent_generator_ensemble_compare.reference=R3 \
  "latent_generator_ensemble_compare.output_dir=$w/latent_generator_ensemble_comparison" "$@"
