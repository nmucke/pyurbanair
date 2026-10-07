#!/bin/bash
# Submit from the repo root. Default: full-corpus training on one 80 GB A100.
#   sbatch job_scripts/delftblue/pretrain_tadpole_ae.slurm train
# Continue from the project checkpoint, or import the old scratch run once:
#   sbatch job_scripts/delftblue/pretrain_tadpole_ae.slurm resume
#   RESUME_FROM=/scratch/$USER/tadpole_ae/train sbatch \
#     job_scripts/delftblue/pretrain_tadpole_ae.slurm resume
# Smoke test on a 10 GB GPU slice (overrides the full-run resources below):
#   sbatch --partition=gpu-a100-small --cpus-per-task=2 --time=00:15:00 \
#     job_scripts/delftblue/pretrain_tadpole_ae.slurm smoke
# Extra arguments are Hydra overrides. Resume mode always forces trainer.resume=true.
# Optional prepared scratch data: PRECHUNK_DATA=true (see README for CPU prep).
#SBATCH --job-name=tadpole_ae
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --account=research-ceg-gse
#SBATCH --output=slurm-tadpole-ae-%j.out
#SBATCH --error=slurm-tadpole-ae-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repo root}"
MODE="${1:-train}"
if [ "$#" -gt 0 ]; then shift; fi
case "${MODE}" in
    smoke|train|resume) ;;
    *) echo "Usage: $0 [smoke|train|resume] [Hydra overrides...]" >&2; exit 2 ;;
esac

# CONFIG — change these or export them before sbatch; tune the AE through Hydra.
DATA_ROOT="${DATA_ROOT:-/projects/urbanair/training_data/pyudales_realistic}"
if [ "${MODE}" = smoke ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-/scratch/${USER}/tadpole_ae/smoke-${SLURM_JOB_ID}}"
else
    # A stable project path preserves checkpoints across 48-hour allocations.
    OUTPUT_DIR="${OUTPUT_DIR:-/projects/urbanair/model_weights/tadpole_ae_b_realistic}"
fi
PIXI_ENV="${PIXI_ENV:-delftblue}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")"

if [ "${MODE}" = resume ]; then
    if [ -s "${OUTPUT_DIR}/checkpoint.pt" ]; then
        echo "Resuming from ${OUTPUT_DIR}/checkpoint.pt"
    else
        RESUME_FROM="${RESUME_FROM:-/scratch/${USER}/tadpole_ae/train}"
        if [ ! -s "${RESUME_FROM}/checkpoint.pt" ] || \
           [ ! -s "${RESUME_FROM}/weights.pt" ] || \
           [ ! -f "${RESUME_FROM}/config.yaml" ]; then
            echo "Resume source needs checkpoint.pt, weights.pt and config.yaml: ${RESUME_FROM}" >&2
            exit 1
        fi
        # A previous import may have copied some files before failing. Reuse
        # matching copies, but never mix artifacts from different runs.
        while IFS= read -r -d '' existing; do
            case "${existing##*/}" in
                weights.pt|config.yaml|best_val.json|metrics.csv|encoder.pt|decoder.pt|geometry_branch.pt) ;;
                *) echo "Unexpected file in resume destination: ${existing}" >&2; exit 1 ;;
            esac
            if [ ! -f "${existing}" ]; then
                echo "Not a regular resume artifact: ${existing}" >&2
                exit 1
            fi
        done < <(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print0)
        echo "Importing training artifacts from ${RESUME_FROM} to ${OUTPUT_DIR}"
        for artifact in weights.pt config.yaml best_val.json \
            metrics.csv encoder.pt decoder.pt geometry_branch.pt; do
            if [ -f "${RESUME_FROM}/${artifact}" ]; then
                if [ -e "${OUTPUT_DIR}/${artifact}" ]; then
                    if ! cmp -s "${RESUME_FROM}/${artifact}" "${OUTPUT_DIR}/${artifact}"; then
                        echo "Resume destination differs from source: ${OUTPUT_DIR}/${artifact}" >&2
                        exit 1
                    fi
                else
                    # Scratch and project storage have different group quotas.
                    cp "${RESUME_FROM}/${artifact}" "${OUTPUT_DIR}/${artifact}"
                fi
            fi
        done
        # The checkpoint appears only after every companion artifact was copied.
        cp "${RESUME_FROM}/checkpoint.pt" \
            "${OUTPUT_DIR}/.checkpoint.pt.importing-${SLURM_JOB_ID}"
        mv "${OUTPUT_DIR}/.checkpoint.pt.importing-${SLURM_JOB_ID}" \
            "${OUTPUT_DIR}/checkpoint.pt"
    fi
fi
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${OUTPUT_DIR}/mplconfig"
export HYDRA_FULL_ERROR=1

# Use the installed Pixi environment; never solve/install on an allocated GPU.
PYTHON=(pixi run --as-is -e "${PIXI_ENV}" python -u)
echo "Tadpole AE: mode=${MODE} job=${SLURM_JOB_ID} host=$(hostname)"
echo "Source: ${DATA_ROOT}; artifacts: ${OUTPUT_DIR}"
srun "${PYTHON[@]}" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(), "CUDA is required"; print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory); assert torch.cuda.is_bf16_supported(), "Select an Ampere-or-newer GPU for bf16"'

OVERRIDES=(
    "model_name=${OUTPUT_DIR}"
    "hydra.run.dir=${OUTPUT_DIR}/hydra"
    trainer.device=cuda
)
if [ "${MODE}" = smoke ]; then
    # Normalization reads entire trajectories, irrespective of time_stride/crops.
    # Copy only two frames of one trajectory per split; never alter the source.
    SMOKE_ROOT="${OUTPUT_DIR}/smoke_data"
    srun "${PYTHON[@]}" - "${DATA_ROOT}" "${SMOKE_ROOT}" <<'PY'
import sys
from pathlib import Path
import xarray as xr

source, target = map(Path, sys.argv[1:])
for split in ("train", "val"):
    files = sorted((source / "state" / split).glob("sample_*.nc"))
    if not files:
        raise RuntimeError(f"No {split} trajectories under {source}")
    dest = target / "state" / split / files[0].name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(files[0]) as ds:
        if ds.sizes["time"] < 2:
            raise RuntimeError(f"Need at least two frames in {files[0]}")
        subset = ds[["u", "v", "w", "blanking"]].isel(time=slice(0, 2))
        subset.to_netcdf(dest)
        print(f"Smoke source: {files[0]} -> {dest}; sizes={dict(subset.sizes)}")
PY
    DATA_ROOT="${SMOKE_ROOT}"
    OVERRIDES+=(
        trainer.num_epochs=1 trainer.lr_warmup_epochs=0 trainer.resume=false
        trainer.checkpoint_every=1 batch_sampler.batch_size=1 batch_sampler.drop_last=false
        dataset.random_crop_size=16 architecture.encoder_crop_size=16
        dataloader.num_workers=0
    )
fi
OVERRIDES+=("dataset.root_dir=${DATA_ROOT}")
if [ "${PRECHUNK_DATA:-false}" = true ]; then
    # The Python entry point completes/revalidates every file before it builds
    # either loader. A smoke run must not reuse the full-corpus cache directory.
    if [ "${MODE}" = smoke ]; then
        PRECHUNK_ROOT="${PRECHUNK_ROOT:-${OUTPUT_DIR}/rechunked_data}"
    else
        PRECHUNK_ROOT="${PRECHUNK_ROOT:-/scratch/${USER}/training_data/$(basename "${DATA_ROOT}")_rechunked}"
    fi
    OVERRIDES+=(prechunk.enabled=true "prechunk.output_root=${PRECHUNK_ROOT}")
fi
RESUME_FLAGS=()
if [ "${MODE}" = resume ]; then
    # A resume request cannot be turned into a fresh run by a Hydra override.
    RESUME_FLAGS=(trainer.resume=true)
fi
srun "${PYTHON[@]}" scripts/neural_surrogate/pretrain_autoencoder.py \
    "${OVERRIDES[@]}" "$@" "${RESUME_FLAGS[@]}"
if [ "${MODE}" = smoke ]; then
    # Exercise loading optimizer/model state, a second epoch, and handoff exports.
    srun "${PYTHON[@]}" scripts/neural_surrogate/pretrain_autoencoder.py \
        "${OVERRIDES[@]}" "$@" trainer.resume=true trainer.num_epochs=2
    srun "${PYTHON[@]}" - "${OUTPUT_DIR}" <<'PY'
import csv
import math
import sys
from pathlib import Path
from omegaconf import OmegaConf

root = Path(sys.argv[1])
cfg = OmegaConf.load(root / "config.yaml")
artifacts = ["weights.pt", "checkpoint.pt", "metrics.csv", "encoder.pt", "decoder.pt"]
if cfg.architecture.get("geometry_branch") is not None:
    artifacts.append("geometry_branch.pt")
for name in artifacts:
    assert (root / name).stat().st_size > 0, name
with (root / "metrics.csv").open() as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 2, f"Expected two epochs, got {len(rows)}"
assert all(math.isfinite(float(row[key])) for row in rows for key in ("train_loss", "val_loss"))
print("Verified two finite training/validation epochs and all handoff artifacts")
PY
    echo "SMOKE PASSED: training, validation, checkpoint resume and AE exports"
fi
echo "Artifacts: ${OUTPUT_DIR}"
