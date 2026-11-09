#!/bin/bash
#   bash job_scripts/delftblue/surrogate_finetune_barcelona.sh [key=value ...]
# Submits the Barcelona fine-tunes of dft_b, ssrolling_tl_roll3_realistic and
# local_fno_realistic on pyudales_barcelona_finetune:
#   1. the whole-frame copy of the data (CPU; validates in seconds when it is
#      already complete). `singleton`: never runs next to another prechunk job.
#   2. the three fine-tunes, once the copy succeeded, one A100 each: 24 h for
#      the two rollout fine-tunes, 12 h for the one-step Local-FNO. Resubmit a
#      single sbatch line to continue a fine-tune from its checkpoint.
# Extra key=value arguments go to every job. Run from the repo root on a
# login node once the data copy is complete.
set -euo pipefail

weights=/projects/urbanair/model_weights
data=/projects/urbanair/training_data/pyudales_barcelona_finetune
frames=/projects/urbanair/training_data/pyudales_barcelona_finetune_frames
jobs=job_scripts/delftblue

if pgrep -u "$USER" -f "^rsync .*pyudales_barcelona_finetune" >/dev/null; then
    echo "rsync of $data is still running: the copy would pin an incomplete source" >&2
    exit 1
fi

# Every job resolves the same source and copy: the manifest pins both.
common=(paths.weights_dir=$weights paths.data_dir=$data prechunk.output_root=$frames "$@")

prep=$(sbatch --parsable --dependency=singleton $jobs/surrogate_prechunk_data.slurm \
    --config-name surrogate/finetune_dft "${common[@]}")
echo "prechunk: $prep"

dft=$(sbatch --parsable --dependency=afterok:$prep --time=24:00:00 \
    $jobs/surrogate_finetune_dft.slurm "${common[@]}" \
    'pretrained_dir=${paths.weights_dir}/dft_b' name=dft_b_ft_barcelona)
ssrolling=$(sbatch --parsable --dependency=afterok:$prep --time=24:00:00 \
    $jobs/surrogate_finetune_ssrolling.slurm "${common[@]}" \
    'pretrained_dir=${paths.weights_dir}/ssrolling_tl_roll3_realistic' \
    name=ssrolling_tl_roll3_realistic_ft_barcelona trainer.num_epochs=100 trainer.patience=10)
local_fno=$(sbatch --parsable --dependency=afterok:$prep \
    $jobs/surrogate_finetune_local_fno.slurm "${common[@]}" \
    'pretrained_dir=${paths.weights_dir}/local_fno_realistic' name=local_fno_realistic_ft_barcelona)
echo "dft_b: $dft, ssrolling: $ssrolling, local_fno: $local_fno"
