#!/usr/bin/env bash
# Run the full single-run filtering (EnKF) pipeline: filter, then metrics, then figures.
#
#   1. scripts/filtering/run_filtering.py            -- runs the filter and saves every artifact.
#   2. scripts/filtering/compute_filtering_metrics.py -- writes run_summary.yaml.
#   3. scripts/filtering/make_filtering_figures.py    -- draws the figures.
#
# The filtering counterpart of scripts/run_esmda_pipeline.sh. All three stages
# share one output directory: conf/run_filtering.yaml's `paths.results_dir`
# (`.temp/filtering_<truth_model>_to_<assim_model>` by default). The runner
# writes there, and the run dir is resolved from the same config -- with any
# overrides applied -- so the metric/figure stages read it back. The output
# location is therefore configured in one place (the YAML), not here.
#
# Stages 2 and 3 produce the same evaluation blocks and figures as the ESMDA
# pipeline -- the parameter marginals (P1), station profiles (S1), sensor fans
# (S5), mean slices (F1) and rank histogram (D1) -- reducing over the filter's
# per-CYCLE ensemble states in place of ESMDA's per-window ones. Which states
# those are depends on one knob:
#
#   run.ensemble_save_on_disk=false  (default) -- only the per-cycle ANALYZED
#       frames survive (state_history.nc), one frame per cycle. Every block still
#       runs, but the per-cycle variance statistic is null and the TKE / <u'w'>
#       moments are taken across cycles rather than within them, so they carry
#       the analysis increments; read those panels as an upper bound.
#   run.ensemble_save_on_disk=true -- the filter keeps every member's full
#       forecast segment under _ensemble_states/cycle_*/, which is the exact
#       analogue of ESMDA's window state files. Costs disk (ensemble x cycles x
#       segment), and buys within-cycle turbulence statistics.
#
# Whichever ran is recorded in run_summary.yaml's `cycle_states` block, so the
# numbers always say which of the two they came from.
#
# Any extra arguments are forwarded to run_filtering.py as Hydra overrides (and
# used to resolve the run dir), e.g.:
#
#   scripts/run_filtering_pipeline.sh filtering.mode=joint filtering.num_cycles=4
#   scripts/run_filtering_pipeline.sh filtering.mode=parameter \
#       filtering/evolution=random_walk filtering/inflation=none
#   scripts/run_filtering_pipeline.sh model@truth_model=pylbm model@assim_model=pylbm
#   scripts/run_filtering_pipeline.sh run.ensemble_save_on_disk=true
set -euo pipefail

cd "$(dirname "$0")/.."

# Resolve the run dir from conf/run_filtering.yaml (paths.results_dir) with the
# same overrides the runner gets, so both agree without pinning a dir here.
RUN_DIR="$(pixi run -e dev python - "$@" <<'PY'
import pathlib
import sys

from hydra import compose, initialize_config_dir

conf_dir = str(pathlib.Path("conf").resolve())
with initialize_config_dir(version_base=None, config_dir=conf_dir):
    cfg = compose(config_name="run_filtering", overrides=sys.argv[1:])
print(cfg.paths.results_dir)
PY
)"
echo "Filtering pipeline output dir: ${RUN_DIR}"

pixi run -e cuda python scripts/filtering/run_filtering.py "$@"
pixi run -e dev python scripts/filtering/compute_filtering_metrics.py --run-dir "${RUN_DIR}"
pixi run -e dev python scripts/filtering/make_filtering_figures.py --run-dir "${RUN_DIR}"

echo "Filtering pipeline complete: ${RUN_DIR}"
