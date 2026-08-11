#!/usr/bin/env bash
# Run the full single-run filter-smoothing pipeline: smooth, then metrics, then figures.
#
#   1. scripts/filter_smoothing/run_filter_smoothing.py            -- runs the method and saves every artifact.
#   2. scripts/filter_smoothing/compute_filter_smoothing_metrics.py -- writes run_summary.yaml.
#   3. scripts/filter_smoothing/make_filter_smoothing_figures.py    -- draws the figures.
#
# The filter-smoothing counterpart of scripts/run_filtering_pipeline.sh and
# scripts/run_esmda_pipeline.sh. All three stages share one output directory:
# conf/run_filter_smoothing.yaml's `paths.results_dir`
# (`.temp/filter_smoothing_<truth_model>_to_<assim_model>` by default). The
# runner writes there, and the run dir is resolved from the same config -- with
# any overrides applied -- so the metric/figure stages read it back. The output
# location is therefore configured in one place (the YAML), not here.
#
# Stages 2 and 3 produce the filtering pipeline's evaluation blocks and figures
# -- station profiles (S1), sensor fans (S5), mean slices (F1), rank histogram
# (D1) -- reducing over the inner filter's per-CYCLE ensemble states, because the
# inner filter IS the EnKF and writes the same per-cycle artifacts. On top of
# those they add the parameter TRAJECTORY blocks (prior vs smoothed posterior vs
# the truth trajectory, per knot) and the OUTER ESMDA loop's convergence
# (iteration_convergence.png, D4), which have no filtering counterpart.
#
# Which per-cycle states the state-side blocks reduce over depends on one knob,
# exactly as in the filtering pipeline:
#
#   run.ensemble_save_on_disk=false  (default) -- only the per-cycle ANALYZED
#       frames survive (state_history.nc), one frame per cycle. Every block still
#       runs, but the per-cycle variance statistic is null and the TKE / <u'w'>
#       moments are taken across cycles rather than within them, so they carry
#       the analysis increments; read those panels as an upper bound.
#   run.ensemble_save_on_disk=true -- the inner filter keeps every member's full
#       forecast segment under _ensemble_states/cycle_*/. Costs disk (ensemble x
#       cycles x segment), and buys within-cycle turbulence statistics.
#
# Whichever ran is recorded in run_summary.yaml's `cycle_states` block, so the
# numbers always say which of the two they came from.
#
# ONE MORE THING THIS PIPELINE HAS TO SAY, and the filtering one does not: with
# `filter_smoothing.num_windows>1` every window's inner filter rewrites the SAME
# cycle_0..cycle_{L-1} artifacts, so only the LAST window's per-cycle states
# survive a T-cycle horizon. run_summary.yaml's `window_layout` block records the
# window geometry and the exact global cycle indices every state/sensor/cycle
# block was scored over; the trajectory blocks cover the whole horizon.
#
# Any extra arguments are forwarded to run_filter_smoothing.py as Hydra overrides
# (and used to resolve the run dir), e.g.:
#
#   scripts/run_filter_smoothing_pipeline.sh filter_smoothing.num_cycles=8 filter_smoothing.num_steps=4
#   scripts/run_filter_smoothing_pipeline.sh filter_smoothing/inner_analysis=letkf \
#       filter_smoothing/inner_localization=distance
#   scripts/run_filter_smoothing_pipeline.sh filter_smoothing.num_windows=10 \
#       filter_smoothing.window_shift=2 filter_smoothing/evolution=random_walk
#   scripts/run_filter_smoothing_pipeline.sh run.ensemble_save_on_disk=true
set -euo pipefail

cd "$(dirname "$0")/.."

# Resolve the run dir from conf/run_filter_smoothing.yaml (paths.results_dir)
# with the same overrides the runner gets, so both agree without pinning a dir
# here.
RUN_DIR="$(pixi run -e dev python - "$@" <<'PY'
import pathlib
import sys

from hydra import compose, initialize_config_dir

conf_dir = str(pathlib.Path("conf").resolve())
with initialize_config_dir(version_base=None, config_dir=conf_dir):
    cfg = compose(config_name="run_filter_smoothing", overrides=sys.argv[1:])
print(cfg.paths.results_dir)
PY
)"
echo "Filter smoothing pipeline output dir: ${RUN_DIR}"

pixi run -e cuda python scripts/filter_smoothing/run_filter_smoothing.py "$@"
pixi run -e dev python scripts/filter_smoothing/compute_filter_smoothing_metrics.py --run-dir "${RUN_DIR}"
pixi run -e dev python scripts/filter_smoothing/make_filter_smoothing_figures.py --run-dir "${RUN_DIR}"

echo "Filter smoothing pipeline complete: ${RUN_DIR}"
