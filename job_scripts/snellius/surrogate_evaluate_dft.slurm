#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_evaluate_dft.slurm ['stepper.sample_indices=[0,1,2]']
# Rolls out the DFT stepper trained by surrogate_train_dft.slurm (dft_b) on test
# trajectories with scripts/surrogate/evaluate_stepper.py on one A100
# (configs/surrogate/eval.yaml, block `stepper`). Outputs in <model dir>/eval.
#SBATCH --job-name=surrogate_evaluate_dft
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
run_gpu python -u scripts/surrogate/evaluate_stepper.py \
  'stepper.models=[${paths.weights_dir}/dft_b]' 'stepper.output_dir=${paths.weights_dir}/dft_b/eval' "$@"
