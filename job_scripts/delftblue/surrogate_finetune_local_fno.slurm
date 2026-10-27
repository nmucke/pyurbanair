#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_finetune_local_fno.slurm pretrained_dir=<model> name=<new name> paths.data_dir=<new corpus> [prechunk.output_root=<copy>]
# Fine-tunes a trained Local-FNO on new data
# (configs/surrogate/baselines/local_fno/finetune.yaml) on one A100, e.g.
# local_fno_realistic on pyudales_barcelona_finetune. Resubmit the same line to
# resume.
#SBATCH --job-name=surrogate_finetune_local_fno
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/baselines/local_fno/finetune trainer.checkpoint_every=1 \
  paths.weights_dir=/projects/urbanair/model_weights \
  paths.training_data_dir=/projects/urbanair/training_data "$@"
