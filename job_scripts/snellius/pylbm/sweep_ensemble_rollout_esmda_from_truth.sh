#!/bin/bash
# Submit a SERIES of rollout-ESMDA-from-truth jobs, ONE SEPARATE SLURM JOB per
# ENSEMBLE SIZE, at a FIXED grid resolution (the k=4 setting, 100 x 80 x 32).
# Every job assimilates against the SAME pre-simulated ground truth -- only the
# ensemble size changes.
#
# The assimilation backend is fixed by the folder this script lives in
# (it submits the sibling rollout_esmda_from_truth.slurm). Run it DIRECTLY from
# the repo root on a Snellius login node:
#
#     bash job_scripts/snellius/<backend>/sweep_ensemble_rollout_esmda_from_truth.sh
#
# Any extra arguments are forwarded as Hydra overrides to EVERY job, e.g.:
#
#     bash job_scripts/snellius/<backend>/sweep_ensemble_rollout_esmda_from_truth.sh esmda.num_steps=4
#
# Each job gets its OWN --time and --cpus-per-task: cores track the ensemble size
# (one core per member, rounded to the partition minimum, partition auto-selected
# by core count). NX/NY/NZ and ENSEMBLE_SIZE are injected via the environment so
# each ensemble size writes to its own RESULTS_DIR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/rollout_esmda_from_truth.slurm"
[ -f "${JOB_SCRIPT}" ] || { echo "error: ${JOB_SCRIPT} not found" >&2; exit 1; }

# Assimilation backend = the folder this script lives in (pyudales|pypalm|pylbm).
MODEL="$(basename "${SCRIPT_DIR}")"

# Set --cpus-per-task to the ensemble size (one core per member) and pick the
# matching Snellius partition. Sets globals PARTITION (rome up to 128 cores/node;
# larger ensembles spill onto genoa, up to 192/node) and CORES_REQ.
size_job() {
  local want="$1"
  CORES_REQ=${want}                     # --cpus-per-task == ensemble size
  if (( want <= 128 )); then
    PARTITION="rome"
  else
    PARTITION="genoa"
    (( want > 192 )) && CORES_REQ=192   # one genoa node holds at most 192 cores
  fi
}

# Fixed grid (k=4, == the 100x80x32 row of the domain sweep).
NX=50
NY=40
NZ=16

# ============================================================================
# Ensemble sizes to sweep, each with its OWN wall clock. With one core per member
# wall-clock is roughly flat, but larger ensembles cost a little more I/O + a
# bigger Kalman solve, so the times grow gently. Sizes above 128 spill onto the
# genoa partition automatically (up to 192 cores).
# Format: "ENSEMBLE_SIZE TIME"   (each row submits one separate job)
# ============================================================================
ENSEMBLE_SIZES=(
  "8     14:00:00"
  "16    14:00:00"
  "32    15:00:00"
  "64    16:00:00"
  # "96    16:00:00"
  # "128   18:00:00"
)

echo "Submitting ${#ENSEMBLE_SIZES[@]} rollout-ESMDA jobs (shared ground truth), one per ensemble size."
echo "Job script: ${JOB_SCRIPT}"
echo "Fixed grid = ${NX}x${NY}x${NZ}; --cpus-per-task tracks the ensemble size per job."
[ "$#" -gt 0 ] && echo "Forwarding extra Hydra overrides to every job: $*"

for row in "${ENSEMBLE_SIZES[@]}"; do
  read -r ens walltime <<<"${row}"
  size_job "${ens}"
  echo "  -> submitting ENSEMBLE_SIZE=${ens}  time=${walltime}  partition=${PARTITION} cpus=${CORES_REQ}"
  sbatch \
    --job-name="rollout_${MODEL}_ens${ens}" \
    --partition="${PARTITION}" \
    --time="${walltime}" \
    --cpus-per-task="${CORES_REQ}" \
    --export="ALL,NX=${NX},NY=${NY},NZ=${NZ},ENSEMBLE_SIZE=${ens}" \
    "${JOB_SCRIPT}" "$@"
done

echo "All jobs submitted -- check with: squeue -u \"\${USER}\""
