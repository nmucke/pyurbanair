#!/bin/bash
#   sbatch job_scripts/delftblue/surrogate_finetune_ssrolling.slurm pretrained_dir=<model> name=<new name> paths.data_dir=<new corpus> [prechunk.output_root=<copy>]
# Fine-tunes a trained SSRollingUrbanNet on new data with Roll-3
# (configs/surrogate/baselines/ssrolling/finetune_roll3.yaml) on one A100, e.g.
# ssrolling_tl_roll3_realistic on pyudales_barcelona_finetune. Every Swin block,
# the encoder and the decoder are recomputed in backward, as for
# ssrolling_tl_roll3_realistic on realistic-size grids (nz 32). Resubmit the
# same line to resume.
#SBATCH --job-name=surrogate_finetune_ssrolling
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/delftblue/out_files/%x-%j.out

source job_scripts/delftblue/env.sh
run python -u scripts/surrogate/train.py --config-name surrogate/baselines/ssrolling/finetune_roll3 trainer.checkpoint_every=1 \
    'trainer.activation_checkpoint_modules=[aurora.backbone.encoder_layers.*.blocks.*,aurora.backbone.decoder_layers.*.blocks.*,aurora.encoder,aurora.decoder]' "$@"
