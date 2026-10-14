#!/usr/bin/env bash
# Run CFD backends side by side on one case, then score and draw them.
#
#   bash workflows/compare_models.sh ['models=[pyudales,pypalm]'] [hydra overrides...]
#
# e.g.
#   bash workflows/compare_models.sh 'models=[pyudales,pypalm]' replicas=2 \
#       +overrides.pypalm.model.forward_model.boundary_condition=inflow_outflow
#
# Runs scripts/comparison/run_models.py with the overrides
# (configs/model_comparison.yaml), then compute_metrics.py and
# visualize_comparison.py on its run dir, <paths.results_dir>. Run it inside the
# dev environment (`pixi shell -e dev`).
set -euo pipefail

cd "$(dirname "$0")/.."
script=scripts/comparison/run_models.py

# The run dir, resolved from the same overrides (Hydra prints the value only).
run_dir=$(python "$script" "$@" --cfg job --resolve -p paths.results_dir | tail -n 1)

python "$script" "$@"
python scripts/comparison/compute_metrics.py "$run_dir"
python scripts/comparison/visualize_comparison.py "$run_dir"
echo "Done: $run_dir"
