#!/bin/bash
# Submit a random-geometry training-data generation that is too long for one
# DelftBlue job as a dependency chain of sharded jobs, producing the same corpus
# as one long single-process run:
#
#   plan  --afterok-->  simulate array (round 0)  --afterany-->  ... resume
#   rounds ...  --afterany-->  finalize
#
# Every resume round resubmits the WHOLE array: shards that finished exit after
# a quick scan, shards that hit the time limit (or had a transient solver
# failure) pick up exactly the samples still missing. Finalize fails loudly,
# naming the shards to resubmit, if anything is still missing after the last
# round.
#
#     OUTPUT_DIR=/projects/urbanair/training_data/pyudales_realistic NUM_SHARDS=64 \
#         bash job_scripts/delftblue/submit_random_geometries_training_data.sh \
#         [hydra overrides for the plan...]
#
# Environment (defaults in brackets):
#   OUTPUT_DIR      dataset root [/projects/urbanair/training_data/pyudales_realistic]
#   NUM_SHARDS      array size [64]. Pick it from the plan log's shard table:
#                   each shard's wall time ~ its share x the single-process time.
#   NCPU            uDALES ranks per shard [16]; must divide every nx (1/2/4/8/16)
#   SHARD_TIME      per-task wall time [24:00:00]
#   RESUME_ROUNDS   extra array passes after the first [1]
#   MAX_CONCURRENT  cap on simultaneously running shards [unlimited]
#   SKIP_PLAN=1     the plan already exists in OUTPUT_DIR (e.g. made on the
#                   login node to inspect the shard table first); submit the
#                   arrays + finalize only
#   DRY_RUN=1       print the sbatch commands instead of submitting
#
# The Hydra overrides you pass only change the PLAN (e.g. training_data.num_train=
# 600); simulate/finalize run the plan's frozen config.yaml. Do not reuse an
# OUTPUT_DIR for a different config -- the plan stage refuses it.

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root: the .slurm file cds to the submit dir

export OUTPUT_DIR="${OUTPUT_DIR:-/projects/urbanair/training_data/pyudales_realistic}"
export NUM_SHARDS="${NUM_SHARDS:-64}"
export NCPU="${NCPU:-16}"
SHARD_TIME="${SHARD_TIME:-24:00:00}"
RESUME_ROUNDS="${RESUME_ROUNDS:-1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"
JOB="job_scripts/delftblue/generate_random_geometries_training_data.slurm"
LOGS="job_scripts/delftblue/out_files"
mkdir -p "${LOGS}"

submit() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "sbatch $*" >&2
        echo "DRYRUN_$RANDOM"
    else
        sbatch --parsable --export=ALL "$@"
    fi
}

echo "Output ${OUTPUT_DIR}: ${NUM_SHARDS} shards x ${NCPU} cores, ${SHARD_TIME}," \
     "1 + ${RESUME_ROUNDS} round(s)"

DEP=()
if [ "${SKIP_PLAN:-0}" != "1" ]; then
    # Pool scan + parameter sampling + two figures: small and quick.
    plan_id=$(submit --job-name=grg_plan --cpus-per-task=2 --mem-per-cpu=4G \
        --time=01:00:00 \
        --output="${LOGS}/slurm-grg_plan-%j.out" --error="${LOGS}/slurm-grg_plan-%j.err" \
        "${JOB}" plan "$@")
    echo "plan:     ${plan_id}"
    DEP=( "--dependency=afterok:${plan_id}" "--kill-on-invalid-dep=yes" )
fi

ARRAY="0-$((NUM_SHARDS - 1))${MAX_CONCURRENT:+%${MAX_CONCURRENT}}"
for round in $(seq 0 "${RESUME_ROUNDS}"); do
    sim_id=$(submit --job-name="grg_sim_r${round}" --array="${ARRAY}" \
        --cpus-per-task="${NCPU}" --time="${SHARD_TIME}" \
        --output="${LOGS}/slurm-grg_sim_r${round}-%A_%a.out" \
        --error="${LOGS}/slurm-grg_sim_r${round}-%A_%a.err" \
        ${DEP[@]+"${DEP[@]}"} "${JOB}" simulate "$@")
    echo "simulate round ${round}: ${sim_id}"
    # afterany: a timed-out shard must still trigger the next (resume) round.
    DEP=( "--dependency=afterany:${sim_id}" )
done

# Opens every sample once for a frame-count check, then plots + 3 animations.
fin_id=$(submit --job-name=grg_finalize --cpus-per-task=4 --mem-per-cpu=8G \
    --time=04:00:00 \
    --output="${LOGS}/slurm-grg_finalize-%j.out" --error="${LOGS}/slurm-grg_finalize-%j.err" \
    "${DEP[@]}" "${JOB}" finalize "$@")
echo "finalize: ${fin_id}"
