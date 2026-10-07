#!/bin/bash
# Evaluate a trained Tadpole AE on DelftBlue. Submit from the repository root:
#   sbatch job_scripts/delftblue/test_autoencoder.slurm
#   MODEL_DIR=/scratch/$USER/tadpole_ae/train sbatch \
#     job_scripts/delftblue/test_autoencoder.slurm split=val num_metric_samples=24
# Extra arguments are Hydra overrides and take precedence over the defaults.
# The training artifact's config.yaml supplies the dataset path and architecture.
#SBATCH --job-name=test_tadpole_ae
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --account=research-ceg-gse
#SBATCH --output=slurm-test-tadpole-ae-%j.out
#SBATCH --error=slurm-test-tadpole-ae-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repo root}"

MODEL_DIR="${MODEL_DIR:-/projects/urbanair/model_weights/tadpole_ae_b_realistic}"
# MODEL_DIR="${MODEL_DIR:-/scratch/ntmucke/tadpole_ae/train}"
PIXI_ENV="${PIXI_ENV:-delftblue}"
for artifact in config.yaml weights.pt; do
    if [ ! -f "${MODEL_DIR}/${artifact}" ]; then
        echo "Missing ${MODEL_DIR}/${artifact}" >&2
        exit 1
    fi
done
MODEL_DIR="$(realpath "${MODEL_DIR}")"
EVAL_DIR="${EVAL_DIR:-${MODEL_DIR}/ae_eval}"
mkdir -p "${EVAL_DIR}"
EVAL_DIR="$(realpath "${EVAL_DIR}")"

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${EVAL_DIR}/mplconfig" HYDRA_FULL_ERROR=1

# Use the installed Pixi environment; do not solve packages on the GPU node.
PYTHON=(pixi run --as-is -e "${PIXI_ENV}" python -u)
echo "Tadpole AE evaluation: job=${SLURM_JOB_ID} host=$(hostname)"
echo "Model: ${MODEL_DIR}; results: ${EVAL_DIR}"
srun "${PYTHON[@]}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print(torch.cuda.get_device_name(0))'
srun "${PYTHON[@]}" scripts/neural_surrogate/test_autoencoder.py \
    "model_dir=${MODEL_DIR}" \
    "output_dir=${EVAL_DIR}" \
    "hydra.run.dir=${EVAL_DIR}/hydra" \
    device=cuda \
    max_internal_batchsize=2 \
    "$@"
echo "Evaluation plots and metrics: ${EVAL_DIR}"
