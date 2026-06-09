#!/bin/bash
# Submit a SERIES of rollout-ESMDA-from-truth jobs, ONE SEPARATE SLURM JOB per
# grid resolution, sweeping NX/NY/NZ from a coarse grid up to the full
# ground-truth resolution. Every job assimilates against the SAME pre-simulated
# ground truth -- only the assimilation grid resolution changes.
#
# The assimilation backend is fixed by the folder this script lives in
# (it submits the sibling rollout_esmda_from_truth.slurm). Run it DIRECTLY from
# the repo root on a Snellius login node:
#
#     bash job_scripts/snellius/<backend>/sweep_domain_rollout_esmda_from_truth.sh
#
# Any extra arguments are forwarded as Hydra overrides to EVERY job, e.g.:
#
#     bash job_scripts/snellius/<backend>/sweep_domain_rollout_esmda_from_truth.sh esmda.num_steps=4
#
# Each job's --time is taken from its row; --cpus-per-task and --partition are
# right-sized from ENSEMBLE_SIZE (one core per ensemble member). NX/NY/NZ and
# ENSEMBLE_SIZE are injected via the environment so each resolution writes to its
# own RESULTS_DIR.
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

# Ensemble size, shared by every job (one core per member -> --cpus-per-task).
ENSEMBLE_SIZE=96

# ============================================================================
# Resolution sweep. The ground truth (job_scripts/snellius/ground_truth.slurm)
# is 100 x 80 x 32, i.e. aspect ratio 25:20:8. Each row keeps that ratio and
# steps from coarse up to the full truth grid, with a per-row wall-clock --time
# that grows with the grid. These walls are rough upper bounds -- tighten/loosen
# per row as you learn the real runtimes.
# Format: "NX NY NZ TIME"   (each row submits one separate job)
# ============================================================================
RESOLUTIONS=(
  "25 20 8     04:00:00"   # k=1  (coarsest)
  "50 40 16    08:00:00"   # k=2
  "75 60 24    16:00:00"   # k=3
  "100 80 32   32:00:00"   # k=4  (== ground-truth resolution)
)

size_job "${ENSEMBLE_SIZE}"
echo "Submitting ${#RESOLUTIONS[@]} rollout-ESMDA jobs (shared ground truth), one per resolution."
echo "Job script: ${JOB_SCRIPT}"
echo "Ensemble size = ${ENSEMBLE_SIZE} -> partition=${PARTITION} --cpus-per-task=${CORES_REQ}."
[ "$#" -gt 0 ] && echo "Forwarding extra Hydra overrides to every job: $*"

for res in "${RESOLUTIONS[@]}"; do
  read -r nx ny nz walltime <<<"${res}"
  echo "  -> submitting NX=${nx} NY=${ny} NZ=${nz}  time=${walltime}  partition=${PARTITION} cpus=${CORES_REQ}"
  sbatch \
    --job-name="rollout_${MODEL}_${nx}x${ny}x${nz}" \
    --partition="${PARTITION}" \
    --time="${walltime}" \
    --cpus-per-task="${CORES_REQ}" \
    --export="ALL,NX=${nx},NY=${ny},NZ=${nz},ENSEMBLE_SIZE=${ENSEMBLE_SIZE}" \
    "${JOB_SCRIPT}" "$@"
done

echo "All jobs submitted -- check with: squeue -u \"\${USER}\""
