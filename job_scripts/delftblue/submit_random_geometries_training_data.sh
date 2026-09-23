#!/bin/bash
# Submit a random-geometry training-data generation that is too long for one
# DelftBlue job as a dependency chain of sharded jobs, producing the same corpus
# as one long single-process run:
#
#   plan  --afterok-->  simulate array (NUM_SHARDS tasks)  --afterany-->  finalize
#
# A simulate task that nears its time limit (or fails) requeues itself under the
# same job id and resumes exactly the samples still missing, up to MAX_REQUEUES
# times -- so only unfinished shards go back in the queue, and finalize (which
# waits for the whole array) starts once the last one is done. Finalize fails
# loudly, naming the shards to resubmit, if anything is still missing then.
#
#     OUTPUT_DIR=/projects/urbanair/training_data/pyudales_realistic NUM_SHARDS=64 \
#         bash job_scripts/delftblue/submit_random_geometries_training_data.sh \
#         [hydra overrides for the plan...]
#
# Environment (defaults in brackets):
#   OUTPUT_DIR      dataset root [/projects/urbanair/training_data/pyudales_realistic]
#   NUM_SHARDS      array size [64]. research-ceg-gse runs at most 64 of this
#                   user's jobs at once, so more shards only queue. The plan log
#                   prints each shard's share of the compute.
#   NCPU            uDALES ranks per shard [16]; must divide every nx (1/2/4/8/16)
#   MEM_PER_CPU     simulate memory per core [3G] (nodes allow ~3.9G/core)
#   SHARD_TIME      per-task wall time before a requeue [12:00:00]. Measured
#                   (Sep 2026, realistic pool, 64 shards x 16 ranks): ~1.6-2.1 us
#                   per cell per simulated second, so shards need ~5-11 h and the
#                   largest geometry group ~7 h. research-ceg-gse's low fairshare
#                   means tasks mostly start via backfill, which favours short
#                   limits -- don't raise this without a reason.
#   MAX_REQUEUES    restarts per simulate task (time limit or failure) [3]
#   MAX_CONCURRENT  cap on simultaneously running shards [unlimited]
#   ACCOUNT         [research-ceg-gse]; innovation allows 1 running / 10 queued
#                   jobs per user, too few for the array
#   PARTITION       [compute-p1,compute-p2]; whichever can start a task first
#   SKIP_PLAN=1     the plan already exists in OUTPUT_DIR (e.g. made on the
#                   login node to inspect the shard table first); submit the
#                   array + finalize only
#   DRY_RUN=1       print the sbatch commands instead of submitting
#
# The Hydra overrides you pass only change the PLAN (e.g. training_data.num_train=
# 600); simulate/finalize run the plan's frozen config.yaml. Do not reuse an
# OUTPUT_DIR for a different config -- the plan stage refuses it. The CODE is
# not frozen: every task imports the checkout as it is when it starts, so leave
# it alone until the run is done.

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root: the .slurm file cds to the submit dir

export OUTPUT_DIR="${OUTPUT_DIR:-/projects/urbanair/training_data/pyudales_realistic}"
export NUM_SHARDS="${NUM_SHARDS:-64}"
export NCPU="${NCPU:-16}"
export MAX_REQUEUES="${MAX_REQUEUES:-3}"
MEM_PER_CPU="${MEM_PER_CPU:-3G}"
SHARD_TIME="${SHARD_TIME:-12:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"
ACCOUNT="${ACCOUNT:-research-ceg-gse}"
PARTITION="${PARTITION:-compute-p1,compute-p2}"
JOB="job_scripts/delftblue/generate_random_geometries_training_data.slurm"
LOGS="job_scripts/delftblue/out_files"
mkdir -p "${LOGS}"

# Fail here rather than after hours in the queue: the tasks run `pixi run
# --as-is`, which never installs, and the pool is gitignored.
for need in .pixi/envs/delftblue libs/pyudales/u-dales/build/release/u-dales; do
    [ -e "${need}" ] || { echo "Missing ${need}: run 'pixi install -e delftblue' first" >&2; exit 1; }
done
if ! ls examples/geometries/processed/*/manifest.csv >/dev/null 2>&1; then
    echo "No geometry pool under examples/geometries/processed/: build it with" \
         "examples/geometries/{download_urbantales_geometries,rasters_to_stl}.py" >&2
    exit 1
fi

submit() {
    local common=( --parsable --export=ALL --account="${ACCOUNT}" --partition="${PARTITION}" )
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "sbatch ${common[*]} $*" >&2
        echo "DRYRUN_$RANDOM"
    else
        sbatch "${common[@]}" "$@"
    fi
}

echo "Output ${OUTPUT_DIR}: ${NUM_SHARDS} shards x ${NCPU} cores, ${SHARD_TIME}" \
     "x up to $((MAX_REQUEUES + 1)) attempts, account ${ACCOUNT}"

DEP=()
if [ "${SKIP_PLAN:-0}" != "1" ]; then
    # Pool scan + parameter sampling + two figures: small and quick.
    plan_id=$(submit --job-name=grg_plan --cpus-per-task=2 --mem-per-cpu=3G \
        --time=01:00:00 \
        --output="${LOGS}/slurm-grg_plan-%j.out" --error="${LOGS}/slurm-grg_plan-%j.err" \
        "${JOB}" plan "$@")
    echo "plan:     ${plan_id}"
    DEP=( "--dependency=afterok:${plan_id}" "--kill-on-invalid-dep=yes" )
fi

ARRAY="0-$((NUM_SHARDS - 1))${MAX_CONCURRENT:+%${MAX_CONCURRENT}}"
sim_id=$(submit --job-name=grg_sim --array="${ARRAY}" \
    --cpus-per-task="${NCPU}" --mem-per-cpu="${MEM_PER_CPU}" --time="${SHARD_TIME}" \
    --output="${LOGS}/slurm-grg_sim-%A_%a.out" \
    --error="${LOGS}/slurm-grg_sim-%A_%a.err" \
    ${DEP[@]+"${DEP[@]}"} "${JOB}" simulate "$@")
echo "simulate: ${sim_id}"

# afterany: finalize must also run (and report the missing shards) when some
# task gave up. Opens every sample once for a frame-count check, then plots +
# 3 animations (GIFs: DelftBlue has no ffmpeg).
fin_id=$(submit --job-name=grg_finalize --cpus-per-task=8 --mem-per-cpu=3G \
    --time=04:00:00 \
    --output="${LOGS}/slurm-grg_finalize-%j.out" --error="${LOGS}/slurm-grg_finalize-%j.err" \
    "--dependency=afterany:${sim_id}" "${JOB}" finalize "$@")
echo "finalize: ${fin_id}"
