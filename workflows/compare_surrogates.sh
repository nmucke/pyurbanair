#!/usr/bin/env bash
# Roll trained surrogates out side by side, then score and draw them.
#
#   bash workflows/compare_surrogates.sh 'models=[dirA,dirB]' [hydra overrides...]
#
# e.g.
#   bash workflows/compare_surrogates.sh 'models=[model_weights/local_fno,model_weights/dft_s]' max_steps=50
#
# Runs scripts/comparison/run_models.py with the overrides (configs/comparison.yaml),
# then compute_metrics.py and visualize_comparison.py on its run dir,
# <paths.results_dir>. Run it inside the dev environment (`pixi shell -e dev`).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 'models=[dirA,dirB]' [hydra overrides...]" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
script=scripts/comparison/run_models.py

# The run dir, resolved from the same overrides (Hydra prints the value only).
run_dir=$(python "$script" "$@" --cfg job --resolve -p paths.results_dir | tail -n 1)

python "$script" "$@"
python scripts/comparison/compute_metrics.py "$run_dir"
python scripts/comparison/visualize_comparison.py "$run_dir"
echo "Done: $run_dir"
