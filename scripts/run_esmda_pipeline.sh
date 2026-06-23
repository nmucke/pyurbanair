#!/usr/bin/env bash
# Run the full single-run ESMDA pipeline: assimilation, then metrics, then figures.
#
#   1. scripts/run_esmda.py             -- runs the DA and saves every artifact.
#   2. scripts/compute_esmda_metrics.py -- writes run_summary.yaml.
#   3. scripts/make_esmda_figures.py    -- draws the figures.
#
# All three stages share one output directory: conf/run_esmda.yaml's
# `paths.results_dir` (`.temp/<truth_model>_to_<assim_model>` by default). The
# runner writes there, and the run dir is resolved from the same config -- with
# any overrides applied -- so the metric/figure stages read it back. The output
# location is therefore configured in one place (the YAML), not here.
#
# Any extra arguments are forwarded to run_esmda.py as Hydra overrides (and used
# to resolve the run dir), e.g.:
#
#   scripts/run_esmda_pipeline.sh esmda/smoother=static \
#       params@prior_params=static params@truth_params=static_truth
#   scripts/run_esmda_pipeline.sh model@truth_model=pylbm model@assim_model=pyudales
set -euo pipefail

cd "$(dirname "$0")/.."

# Resolve the run dir from conf/run_esmda.yaml (paths.results_dir) with the same
# overrides the runner gets, so both agree without pinning a dir here.
RUN_DIR="$(pixi run -e dev python - "$@" <<'PY'
import pathlib
import sys

from hydra import compose, initialize_config_dir

conf_dir = str(pathlib.Path("conf").resolve())
with initialize_config_dir(version_base=None, config_dir=conf_dir):
    cfg = compose(config_name="run_esmda", overrides=sys.argv[1:])
print(cfg.paths.results_dir)
PY
)"
echo "ESMDA pipeline output dir: ${RUN_DIR}"

pixi run -e cuda python scripts/run_esmda.py "$@"
pixi run -e dev python scripts/compute_esmda_metrics.py --run-dir "${RUN_DIR}"
pixi run -e dev python scripts/make_esmda_figures.py --run-dir "${RUN_DIR}"

echo "ESMDA pipeline complete: ${RUN_DIR}"
