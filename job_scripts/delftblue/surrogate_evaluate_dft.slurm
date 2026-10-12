#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_evaluate_dft.slurm ['stepper.sample_indices=[0,1,2]']
# Rolls out the DFT stepper trained by surrogate_train_dft.slurm (dft_b) on test
# trajectories with scripts/surrogate/evaluate_stepper.py on one A100
# (configs/surrogate/eval.yaml, block `stepper`). Outputs in <model dir>/eval.
#SBATCH --job-name=surrogate_evaluate_dft
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/evaluate_stepper.py \
  'stepper.models=[${paths.weights_dir}/dft_b]' 'stepper.output_dir=${paths.weights_dir}/dft_b/eval' "$@"
