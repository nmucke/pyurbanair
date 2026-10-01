#!/usr/bin/env bash
# Run an assimilation, then compute its metrics and draw its figures.
#
#   bash workflows/assimilation_workflow.sh <smoother|filtering|hybrid> [hydra overrides...]
#
# e.g.
#   bash workflows/assimilation_workflow.sh smoother params@prior_params=dynamic
#   bash workflows/assimilation_workflow.sh filtering 'filtering.analysis=${analysis.letkf}'
#
# Runs scripts_new/run_<method>.py with the overrides, then compute_metrics.py
# and visualize_assimilation.py on its run dir, <paths.results_dir>/<method>.
# Run it inside the dev environment (`pixi shell -e dev`).
set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^(smoother|filtering|hybrid)$ ]]; then
    echo "usage: $0 <smoother|filtering|hybrid> [hydra overrides...]" >&2
    exit 1
fi
method=$1
shift

cd "$(dirname "$0")/.."
script=scripts_new/run_${method}.py

# The run dir, resolved from the same overrides (Hydra prints the value only).
results_dir=$(python "$script" "$@" --cfg job --resolve -p paths.results_dir | tail -n 1)
run_dir=$results_dir/$method

python "$script" "$@"
python scripts_new/compute_metrics.py "$run_dir"
python scripts_new/visualize_assimilation.py "$run_dir"
echo "Done: $run_dir"
