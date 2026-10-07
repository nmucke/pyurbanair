#!/usr/bin/env bash
# Run the forward model, then draw its figures and its HTML viewer.
#
#   bash workflows/forward_workflow.sh [hydra overrides...]
#
# e.g.
#   bash workflows/forward_workflow.sh model=pylbm params=static_truth
#
# Runs scripts/run_forward.py with the overrides, then visualize_forward.py
# and the HTML viewer (python -m visualization, in the `rendering` pixi env for
# its 3D view) on its run dir, <paths.results_dir>. The viewer bundle goes to
# <run dir>/viewer, replacing an earlier one; serve it with the command printed
# at the end.
# Run it inside the dev environment (`pixi shell -e dev`).
set -euo pipefail

cd "$(dirname "$0")/.."
script=scripts/run_forward.py

# The run dir, resolved from the same overrides (Hydra prints the value only).
run_dir=$(python "$script" "$@" --cfg job --resolve -p paths.results_dir | tail -n 1)

python "$script" "$@"
python scripts/visualize_forward.py "$run_dir"
rm -rf "$run_dir/viewer"
pixi run -e rendering python -m visualization "$run_dir" "$run_dir/viewer"
echo "Done: $run_dir"
echo "Viewer: python -m visualization --serve $run_dir/viewer"
