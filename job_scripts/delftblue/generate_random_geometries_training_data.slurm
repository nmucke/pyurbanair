#!/bin/bash
# ONE stage of a sharded random-geometry training-data generation on DelftBlue
# (scripts/neural_surrogate/generate_random_geometries_training_data.py, pyudales).
# Normally submitted by submit_random_geometries_training_data.sh, which chains
# plan -> simulate array (+ resume rounds) -> finalize with SLURM dependencies.
# By hand:
#
#     export OUTPUT_DIR=/projects/urbanair/training_data/pyudales_realistic NUM_SHARDS=64
#     sbatch generate_random_geometries_training_data.slurm plan
#     sbatch --array=0-63 generate_random_geometries_training_data.slurm simulate
#     sbatch generate_random_geometries_training_data.slurm finalize
#
# Stages (see training_data.sharding in conf/neural_surrogate/training_data.yaml):
#   plan      draws every geometry assignment + parameter trajectory ONCE and
#             freezes them, with the resolved config, in OUTPUT_DIR
#   simulate  array task i runs shard i of NUM_SHARDS against that frozen plan;
#             samples already on disk are skipped, so resubmitting a task that
#             hit the time limit resumes it
#   finalize  checks all samples are on disk, writes params.nc + figures
#
# Environment:
#   OUTPUT_DIR  (required) dataset root; must be the same for every stage
#   NUM_SHARDS  (required) must be the same for every stage
#   NCPU        uDALES MPI ranks per shard (default: cpus-per-task). Must divide
#               every sampled nx; all pool nx are multiples of 16, so 1/2/4/8/16.
# Extra Hydra overrides on the sbatch line are passed through. For simulate and
# finalize only runtime keys take effect (scratch paths, ncpu, verbose): the
# physics comes from the plan's frozen config.yaml, whatever conf/ says now.
#
#SBATCH --job-name=gen_rand_geom
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-%x-%A_%a.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-%x-%A_%a.err

set -euo pipefail

STAGE="${1:?usage: sbatch $0 <plan|simulate|finalize> [hydra overrides...]}"
shift
: "${OUTPUT_DIR:?set OUTPUT_DIR (the dataset root, shared by every stage)}"
: "${NUM_SHARDS:?set NUM_SHARDS (shared by every stage)}"
NCPU="${NCPU:-${SLURM_CPUS_PER_TASK:-1}}"

SHARD_FLAGS=()
if [ "${STAGE}" = "simulate" ]; then
    SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:?simulate runs as an array task (or set SHARD_INDEX)}}"
    SHARD_FLAGS+=( "training_data.sharding.shard_index=${SHARD_INDEX}" )
fi

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p job_scripts/delftblue/out_files "${OUTPUT_DIR}"

# Per-geometry solver scratch goes on /scratch (beegfs), private to this task
# (SLURM_JOB_ID is unique per array task). Cleaned on success; left behind on
# failure for inspection.
RUN_TEMP_DIR="/scratch/${USER}/urbanair_temp/${SLURM_JOB_ID:-$$}"
mkdir -p "${RUN_TEMP_DIR}"
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "DELFTBLUE random-geometry training data: stage=${STAGE}" \
     "shard=${SHARD_INDEX:--}/${NUM_SHARDS} job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Output: ${OUTPUT_DIR}  (temp: ${RUN_TEMP_DIR}, ncpu=${NCPU})"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# DelftBlue MPI environment: force ob1/tcp so a stale pixi cache or unsourced
# activation can't pull UCX/InfiniBand into the MPI stack and crash
# MPI_Finalize. Unlike Snellius, DelftBlue's OpenMPI DOES need osc=pt2pt.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# The solver owns the parallelism (NCPU MPI ranks); keep BLAS single-threaded.
# Line-buffer Python so a crash keeps its traceback.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# hydra.run.dir is absolute and per task: the default lives under .temp, which
# the other DelftBlue submit files also avoid (see generate_training_data.slurm).
pixi run -e delftblue -- python -u \
    scripts/neural_surrogate/generate_random_geometries_training_data.py \
    "training_data.output_dir=${OUTPUT_DIR}" \
    "training_data.sharding.stage=${STAGE}" \
    "training_data.sharding.num_shards=${NUM_SHARDS}" \
    ${SHARD_FLAGS[@]+"${SHARD_FLAGS[@]}"} \
    "hydra.run.dir=${OUTPUT_DIR}/hydra/${STAGE}_${SLURM_JOB_ID:-$$}" \
    "paths.experiment_dir=${RUN_TEMP_DIR}" \
    "model.forward_model.ncpu=${NCPU}" \
    "$@"

echo "Done -- stage ${STAGE} (${OUTPUT_DIR})"
