#!/bin/bash
#   sbatch job_scripts/snellius/surrogate_finetune_ssrolling.slurm [finetune_roll3|finetune_roll3_lora] pretrained_dir=<model> name=<new name> paths.data_dir=<new corpus> [prechunk.output_root=<copy>]
# Fine-tunes a trained SSRollingUrbanNet on new data with Roll-3
# (configs/surrogate/baselines/ssrolling/finetune_roll3.yaml) on one A100, e.g.
# ssrolling_tl_roll3_realistic on pyudales_barcelona_finetune; every weight,
# or with finetune_roll3_lora LoRA adapters. Every Swin block,
# the encoder and the decoder are recomputed in backward, as for
# ssrolling_tl_roll3_realistic on realistic-size grids (nz 32). Resubmit the
# same line to resume.
#SBATCH --job-name=surrogate_finetune_ssrolling
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei19851
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=job_scripts/snellius/out_files/%x-%j.out

stage=finetune_roll3
if [[ "${1:-}" =~ ^finetune_roll3(_lora)?$ ]]; then
    stage=$1
    shift
fi

source job_scripts/snellius/env.sh
# The snellius env has CPU-only torch (+cpu); train in the CUDA env
# (`pixi install -e cuda` once on a login node).
run_gpu() { pixi run -e cuda -- "$@"; }
run_gpu python -u scripts/surrogate/train.py --config-name "surrogate/baselines/ssrolling/${stage}" trainer.checkpoint_every=1 \
    'trainer.activation_checkpoint_modules=[aurora.backbone.encoder_layers.*.blocks.*,aurora.backbone.decoder_layers.*.blocks.*,aurora.encoder,aurora.decoder]' \
    paths.weights_dir=/projects/prjs2304/urbanair/model_weights \
    paths.training_data_dir=/projects/prjs2304/urbanair/training_data "$@"
