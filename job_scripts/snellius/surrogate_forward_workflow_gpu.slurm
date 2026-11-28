#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_forward_workflow_gpu.slurm geometry.stl_path=<stl> domain.nx=... [overrides...]
# Runs workflows/surrogate_forward_workflow.sh (the neural surrogate on any STL,
# then its figures and HTML viewer) on one H100 (94 GB: a city-scale grid in
# full precision needs more than an A100's 40 GB; add
# model.forward_model.amp_dtype=bfloat16 to roughly halve it).
#SBATCH --job-name=surrogate_forward_workflow_gpu
#SBATCH --partition=gpu_h100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); run in the CUDA env
# (`pixi install -e cuda` once on a login node).
pixi run -e cuda -- bash workflows/surrogate_forward_workflow.sh "$@"
