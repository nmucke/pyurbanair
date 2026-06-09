#!/bin/bash
# Shared DELFTBLUE sweep engine for rollout-ESMDA-from-truth. ONE implementation,
# used by every backend, so all three models sweep the IDENTICAL set of values
# and run the exact same experiment at each point -- only the assimilation solver
# differs.
#
# DelftBlue sibling of job_scripts/local/sweep_base.sh. The difference: instead of
# running each point sequentially in this shell, it SUBMITS one SLURM job per
# swept value (sbatch). Each job's --cpus-per-task is sized to the ensemble size
# (one core per ensemble member -- NO parallel-worker cap), capped at a single
# 64-core compute node; ensembles above 64 still run with num_parallel == ensemble
# (oversubscribed, which PALM in particular tolerates). Each row carries its own
# --time. --mem-per-cpu comes from each runner's own SBATCH header (backend-fixed).
#
# Not run directly: each backend folder carries three thin wrappers
# (sweep_{domain,ensemble,esmda_steps}_rollout_esmda_from_truth.sh) that call
# this with the sweep kind and the sibling .slurm runner:
#
#     bash sweep_base.sh <domain|ensemble|steps> <path/to/rollout_esmda_from_truth.slurm> [hydra overrides...]
#
# The swept value is injected into each job via --export (NX/NY/NZ, ENSEMBLE_SIZE
# or NUM_ESMDA_STEPS) so it lands in its own RESULTS_DIR. Any extra arguments are
# forwarded as Hydra overrides to EVERY job.
set -uo pipefail

KIND="${1:?usage: sweep_base.sh <domain|ensemble|steps> <runner.slurm> [hydra overrides...]}"
RUNNER="${2:?usage: sweep_base.sh <domain|ensemble|steps> <runner.slurm> [hydra overrides...]}"
shift 2
[ -f "${RUNNER}" ] || { echo "error: runner '${RUNNER}' not found" >&2; exit 1; }
MODEL="$(basename "$(dirname "${RUNNER}")")"

# ============================================================================
# Canonical sweep value lists -- defined ONCE here so every backend sweeps the
# identical set (matches job_scripts/local/sweep_base.sh). Each row carries a
# per-job wall clock --time (<=24h, the DelftBlue compute limit); tighten/loosen
# as you learn the real runtimes.
# ============================================================================
# Resolution sweep (domain): "NX NY NZ TIME" rows. The ground truth is
# 100 x 80 x 32 (aspect ratio 25:20:8); each row keeps that ratio, coarse ->
# ground-truth grid.
RESOLUTIONS=(
  "25 20 8     02:00:00"   # k=1  (coarsest)
  "50 40 16    04:00:00"   # k=2
  # "75 60 24    12:00:00"   # k=3
  # "100 80 32   24:00:00"   # k=4  (== ground-truth resolution)
)
# Ensemble-size sweep: "ENSEMBLE_SIZE TIME", at the fixed grid below.
ENSEMBLE_SIZES=(
  "8     08:00:00"
  "16    08:00:00"
  "32    10:00:00"
  "64    12:00:00"
  "96    16:00:00"
)
# ESMDA-steps sweep: "NUM_ESMDA_STEPS TIME", at the fixed grid + ensemble below.
ESMDA_STEPS=(
  "1   04:00:00"
  "2   08:00:00"
  "3   12:00:00"
  "4   16:00:00"
)

# Fixed values for the dimensions a given sweep holds constant.
FIXED_NX="${FIXED_NX:-75}"
FIXED_NY="${FIXED_NY:-60}"
FIXED_NZ="${FIXED_NZ:-24}"
FIXED_ENSEMBLE_SIZE="${FIXED_ENSEMBLE_SIZE:-96}"
FIXED_NUM_ESMDA_STEPS="${FIXED_NUM_ESMDA_STEPS:-3}"
# ============================================================================

# Size the SLURM allocation from the ensemble size: --cpus-per-task == ensemble
# size (one core per ensemble member, NO parallel cap), capped at one 64-core
# compute node. Ensembles above 64 still run num_parallel == ensemble
# (oversubscribed). Sets globals PARTITION and CORES_REQ.
size_job() {
  local want="$1" node_max=64
  PARTITION="compute"
  CORES_REQ=${want}
  if (( CORES_REQ > node_max )); then
    echo "warning: ensemble_size=${want} exceeds one compute node (${node_max} cores);" >&2
    echo "         requesting ${node_max} cores; the run still uses num_parallel=${want} (oversubscribed)." >&2
    CORES_REQ=${node_max}
  fi
}

# Hydra overrides forwarded to every job.
FORWARD=( "$@" )

# Submit one job. Args: jobname, walltime, then KEY=VALUE pairs to --export.
submit_one() {
  local jobname="$1" walltime="$2"; shift 2
  local export_str="ALL"
  local kv
  for kv in "$@"; do export_str="${export_str},${kv}"; done
  echo "  -> ${jobname}  time=${walltime}  partition=${PARTITION} cpus=${CORES_REQ}"
  sbatch \
    --job-name="${jobname}" \
    --partition="${PARTITION}" \
    --time="${walltime}" \
    --cpus-per-task="${CORES_REQ}" \
    --export="${export_str}" \
    "${RUNNER}" "${FORWARD[@]}"
}

case "${KIND}" in
  domain)
    size_job "${FIXED_ENSEMBLE_SIZE}"
    echo "DELFTBLUE [${MODEL}] DOMAIN sweep -- ${#RESOLUTIONS[@]} resolutions (ensemble=${FIXED_ENSEMBLE_SIZE}, steps=${FIXED_NUM_ESMDA_STEPS})."
    [ "${#FORWARD[@]}" -gt 0 ] && echo "Forwarding to every job: ${FORWARD[*]}"
    for res in "${RESOLUTIONS[@]}"; do
      read -r nx ny nz walltime <<<"${res}"
      submit_one "rollout_${MODEL}_${nx}x${ny}x${nz}" "${walltime}" \
        "NX=${nx}" "NY=${ny}" "NZ=${nz}" \
        "ENSEMBLE_SIZE=${FIXED_ENSEMBLE_SIZE}" "NUM_ESMDA_STEPS=${FIXED_NUM_ESMDA_STEPS}"
    done
    ;;
  ensemble)
    echo "DELFTBLUE [${MODEL}] ENSEMBLE sweep -- ${#ENSEMBLE_SIZES[@]} sizes (grid=${FIXED_NX}x${FIXED_NY}x${FIXED_NZ}, steps=${FIXED_NUM_ESMDA_STEPS})."
    [ "${#FORWARD[@]}" -gt 0 ] && echo "Forwarding to every job: ${FORWARD[*]}"
    for row in "${ENSEMBLE_SIZES[@]}"; do
      read -r ens walltime <<<"${row}"
      size_job "${ens}"
      submit_one "rollout_${MODEL}_ens${ens}" "${walltime}" \
        "NX=${FIXED_NX}" "NY=${FIXED_NY}" "NZ=${FIXED_NZ}" \
        "ENSEMBLE_SIZE=${ens}" "NUM_ESMDA_STEPS=${FIXED_NUM_ESMDA_STEPS}"
    done
    ;;
  steps)
    size_job "${FIXED_ENSEMBLE_SIZE}"
    echo "DELFTBLUE [${MODEL}] ESMDA-STEPS sweep -- ${#ESMDA_STEPS[@]} step counts (grid=${FIXED_NX}x${FIXED_NY}x${FIXED_NZ}, ensemble=${FIXED_ENSEMBLE_SIZE})."
    [ "${#FORWARD[@]}" -gt 0 ] && echo "Forwarding to every job: ${FORWARD[*]}"
    for row in "${ESMDA_STEPS[@]}"; do
      read -r steps walltime <<<"${row}"
      submit_one "rollout_${MODEL}_steps${steps}" "${walltime}" \
        "NX=${FIXED_NX}" "NY=${FIXED_NY}" "NZ=${FIXED_NZ}" \
        "ENSEMBLE_SIZE=${FIXED_ENSEMBLE_SIZE}" "NUM_ESMDA_STEPS=${steps}"
    done
    ;;
  *)
    echo "error: unknown sweep kind '${KIND}' (expected domain|ensemble|steps)" >&2
    exit 1
    ;;
esac

echo
echo "[${MODEL}] ${KIND} sweep submitted -- check with: squeue -u \"\${USER}\""
