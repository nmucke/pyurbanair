#!/bin/bash
# Submit from the repo root. Default: two tiny epochs, including checkpoint resume.
#   sbatch job_scripts/delftblue/pretrain_tadpole_ae.slurm smoke
# Full corpus (edit resources/hyperparameters first):
#   sbatch --partition=gpu-a100 --cpus-per-task=8 --mem-per-cpu=4G --time=04:00:00 \
#     job_scripts/delftblue/pretrain_tadpole_ae.slurm train trainer.num_epochs=200
# Extra arguments are Hydra overrides and take precedence over the defaults.
#SBATCH --job-name=tadpole_ae
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:15:00
#SBATCH --account=research-ceg-gse
#SBATCH --output=slurm-tadpole-ae-%j.out
#SBATCH --error=slurm-tadpole-ae-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repo root}"
MODE="${1:-smoke}"
if [ "$#" -gt 0 ]; then shift; fi
case "${MODE}" in smoke|train) ;; *) echo "Usage: $0 [smoke|train] [Hydra overrides...]" >&2; exit 2 ;; esac

# CONFIG — change these or export them before sbatch; tune the AE through Hydra.
DATA_ROOT="${DATA_ROOT:-/projects/urbanair/training_data/pyudales_realistic}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/${USER}/tadpole_ae/${MODE}-${SLURM_JOB_ID}}"
PIXI_ENV="${PIXI_ENV:-delftblue}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")"
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
    trainer.checkpoint_every=1
    dataloader.num_workers=1
    batch_sampler.batch_size=1
    batch_sampler.drop_last=false
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
        dataset.random_crop_size=16 architecture.encoder_crop_size=16
        dataloader.num_workers=0
    )
fi
OVERRIDES+=("dataset.root_dir=${DATA_ROOT}")
srun "${PYTHON[@]}" scripts/neural_surrogate/pretrain_autoencoder.py "${OVERRIDES[@]}" "$@"
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
