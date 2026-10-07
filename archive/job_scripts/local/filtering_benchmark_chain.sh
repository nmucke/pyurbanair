#!/usr/bin/env bash
# Wait for the shared truth artifact, then start the benchmark campaign.
#
# Exists so the whole night is one detached chain: truth generation and the
# campaign are separate processes, and nothing should require a human (or an
# agent) to be connected in between.
#
# WORKERS=4 with NUM_LANES=3 puts 12 single-core members on the box at once.
# ensemble.num_cpus_per_process stays 1 in the driver: uDALES with ncpu>1
# mis-stitches its per-rank output, so member-level parallelism (forkserver, one
# core per member) is the ONLY safe kind here. 12 is deliberately short of
# 3x8=24 -- this 3950X is DRAM-bandwidth-bound past ~4-8 concurrent members, so
# 24 would thrash rather than go faster, and would starve the other long-running
# job on this machine.
set -uo pipefail
cd "$(dirname "$0")/../.."

BENCH_ROOT="${BENCH_ROOT:-.temp/filtering_state_reduction_benchmark}"
LOGS="${BENCH_ROOT}/_logs"

echo "$(date -Is) CHAIN waiting for truth generation"
until grep -q "FULLTRUTH_EXIT" "${LOGS}/truth_gen.log" 2>/dev/null; do sleep 20; done
if ! grep -q "FULLTRUTH_EXIT=0" "${LOGS}/truth_gen.log"; then
  echo "$(date -Is) CHAIN ABORT: truth generation failed; campaign not started" >&2
  exit 1
fi
echo "$(date -Is) CHAIN truth ready:"
grep -E "Saved truth" "${LOGS}/truth_gen.log" || true

export TRUTH_DIR="${PWD}/${BENCH_ROOT}/_truth"
export BENCH_ROOT
export NUM_LANES="${NUM_LANES:-3}"
export WORKERS="${WORKERS:-4}"
export NUM_CYCLES="${NUM_CYCLES:-60}"
export ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-50}"

exec bash job_scripts/local/filtering_state_reduction_benchmark.sh
