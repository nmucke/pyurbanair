#!/bin/bash
# Submit a SERIES of rollout-ESMDA-from-truth jobs, ONE SEPARATE SLURM JOB per
# NUMBER OF ESMDA STEPS (iterations per assimilation window), at a FIXED grid
# resolution (the k=4 setting, 100 x 80 x 32) and FIXED ensemble size (96).
# Every job assimilates against the SAME pre-simulated ground truth -- only
# esmda.num_steps changes.
#
# The assimilation backend is fixed by the folder this script lives in
# (it submits the sibling rollout_esmda_from_truth.slurm). Run it DIRECTLY from
# the repo root on a Snellius login node:
#
#     bash job_scripts/snellius/<backend>/sweep_esmda_steps_rollout_esmda_from_truth.sh
#
# Any extra arguments are forwarded as Hydra overrides to EVERY job, e.g.:
#
#     bash job_scripts/snellius/<backend>/sweep_esmda_steps_rollout_esmda_from_truth.sh esmda.seed=1
#
# Each job's --time is taken from its row; --cpus-per-task and --partition are
# right-sized from ENSEMBLE_SIZE (one core per ensemble member). NX/NY/NZ,
# ENSEMBLE_SIZE and NUM_ESMDA_STEPS are injected via the environment so each
# value writes to its own RESULTS_DIR.
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

# Fixed grid (k=4, == the 100x80x32 row of the domain sweep) and fixed ensemble
# size (one core per member -> --cpus-per-task).
NX=100
NY=80
NZ=32
ENSEMBLE_SIZE=96

# ============================================================================
# ESMDA step counts to sweep. Cost scales ~linearly with the step count (each
# extra step is another full ensemble forward solve per window), so the wall
# clock grows with it -- each row carries its own --time. These walls are rough
# upper bounds; tighten as you learn the real runtimes.
# Format: "NUM_ESMDA_STEPS TIME"   (each row submits one separate job)
# ============================================================================
ESMDA_STEPS=(
  "1   08:00:00"
  "2   16:00:00"
  "4   24:00:00"
  "8   48:00:00"
)

size_job "${ENSEMBLE_SIZE}"
echo "Submitting ${#ESMDA_STEPS[@]} rollout-ESMDA jobs (shared ground truth), one per step count."
echo "Job script: ${JOB_SCRIPT}"
echo "Fixed grid = ${NX}x${NY}x${NZ}, ensemble = ${ENSEMBLE_SIZE} -> partition=${PARTITION} --cpus-per-task=${CORES_REQ}."
[ "$#" -gt 0 ] && echo "Forwarding extra Hydra overrides to every job: $*"

for row in "${ESMDA_STEPS[@]}"; do
  read -r steps walltime <<<"${row}"
  echo "  -> submitting NUM_ESMDA_STEPS=${steps}  time=${walltime}  partition=${PARTITION} cpus=${CORES_REQ}"
  sbatch \
    --job-name="rollout_${MODEL}_steps${steps}" \
    --partition="${PARTITION}" \
    --time="${walltime}" \
    --cpus-per-task="${CORES_REQ}" \
    --export="ALL,NX=${NX},NY=${NY},NZ=${NZ},ENSEMBLE_SIZE=${ENSEMBLE_SIZE},NUM_ESMDA_STEPS=${steps}" \
    "${JOB_SCRIPT}" "$@"
done

echo "All jobs submitted -- check with: squeue -u \"\${USER}\""
