#!/bin/bash
# Thin wrapper: LOCAL sweep over ESMDA step count (iterations per window) for the pyudales backend.
#
# Delegates to the shared engine ../sweep_base.sh, which defines the canonical
# swept values ONCE so every backend runs the IDENTICAL sweep -- only the
# assimilation solver (this folder's rollout_esmda_from_truth.sh) differs. Runs
# sequentially in this shell; a single failing point does not abort the rest.
#
# Run from anywhere:
#
#     bash job_scripts/local/pyudales/sweep_esmda_steps_rollout_esmda_from_truth.sh
#
# Any extra arguments are forwarded as Hydra overrides to EVERY run, e.g.:
#
#     bash job_scripts/local/pyudales/sweep_esmda_steps_rollout_esmda_from_truth.sh esmda.seed=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../sweep_base.sh" steps "${SCRIPT_DIR}/rollout_esmda_from_truth.sh" "$@"
