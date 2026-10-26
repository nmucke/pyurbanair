#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_finetune_local_fno.slurm pretrained_dir=<model> name=<new name> paths.data_dir=<new corpus> [prechunk.output_root=<copy>]
# Fine-tunes a trained Local-FNO on new data
# (configs/surrogate/baselines/local_fno/finetune.yaml) on one A100, e.g.
# local_fno_realistic on pyudales_barcelona_finetune. Resubmit the same line to
# resume.
#SBATCH --job-name=surrogate_finetune_local_fno
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name surrogate/baselines/local_fno/finetune trainer.checkpoint_every=1 "$@"
