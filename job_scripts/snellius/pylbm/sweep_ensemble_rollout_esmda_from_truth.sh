#!/bin/bash
# Thin wrapper: SNELLIUS sweep over ensemble size for the pylbm backend.
#
# Delegates to the shared engine ../sweep_base.sh, which defines the canonical
# swept values ONCE so every backend runs the IDENTICAL sweep -- only the
# assimilation solver (this folder's rollout_esmda_from_truth.slurm) differs. It
# SUBMITS one SLURM job per swept value (cores == ensemble size, no parallel cap).
#
# Run from a Snellius login node:
#
#     bash job_scripts/snellius/pylbm/sweep_ensemble_rollout_esmda_from_truth.sh
#
# Any extra arguments are forwarded as Hydra overrides to EVERY job, e.g.:
#
#     bash job_scripts/snellius/pylbm/sweep_ensemble_rollout_esmda_from_truth.sh esmda.seed=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../sweep_base.sh" ensemble "${SCRIPT_DIR}/rollout_esmda_from_truth.slurm" "$@"
