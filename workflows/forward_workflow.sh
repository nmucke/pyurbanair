#!/usr/bin/env bash
# Run the forward model, then draw its figures.
#
#   bash workflows/forward_workflow.sh [hydra overrides...]
#
# e.g.
#   bash workflows/forward_workflow.sh model=pylbm params=static_truth
#
# Runs scripts/run_forward.py with the overrides, then visualize_forward.py
# on its run dir, <paths.results_dir>.
# Run it inside the dev environment (`pixi shell -e dev`).
set -euo pipefail

cd "$(dirname "$0")/.."
script=scripts/run_forward.py

# The run dir, resolved from the same overrides (Hydra prints the value only).
run_dir=$(python "$script" "$@" --cfg job --resolve -p paths.results_dir | tail -n 1)

python "$script" "$@"
python scripts/visualize_forward.py "$run_dir"
echo "Done: $run_dir"
