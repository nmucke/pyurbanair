#!/usr/bin/env bash
# Run the full single-run filtering (EnKF) pipeline: filter, then metrics, then figures.
#
#   1. scripts/filtering/run_filtering.py            -- runs the filter and saves every artifact.
#   2. scripts/filtering/compute_filtering_metrics.py -- writes run_summary.yaml.
#   3. scripts/filtering/make_filtering_figures.py    -- draws the figures.
#   4. scripts/esmda/compute_esmda_metrics.py + make_esmda_figures.py, run over
#      the WINDOW artifacts the filter writes in the ESMDA schema, into
#      <run dir>/esmda_view/ (see "The ESMDA-schema view" below).
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
# The ESMDA-schema view (`<run dir>/esmda_view/`)
# ----------------------------------------------
# A windowed filtering run also writes the ESMDA artifact set -- windows/,
# posterior_state_mean.nc, the assembled parameter files, truth_access.yaml --
# so the ESMDA metric/figure stages run on it unchanged and give a filtering run
# the same panels an ESMDA run gets (including D3, the observation-space data
# mismatch, which the filtering stages have no counterpart for). Both families
# write `run_summary.yaml`, `eval_fields.nc` and same-named figures, and they
# describe DIFFERENT binnings -- the ESMDA stages bin by window, the filtering
# stages by cycle -- so running both into one directory would leave whichever
# ran last, silently.
#
# So they do not share one: `esmda_view/` holds symlinks to the artifacts both
# families read (config.yaml, truth_access.yaml, run_info.yaml, the root *.nc,
# windows/) and is where the ESMDA stages write their own summary, eval fields
# and figures. Nothing is copied and nothing is renamed: the run dir's root
# layout is exactly what it was, which is what the sweep tooling
# (scripts/figure_creation/compare_sweep_results.py globs */run_summary.yaml)
# and the filtering stages read, and stage 4's outputs sit one directory down.
# The two summaries' `sensor_statistics[*].n_windows` is the tell: W in the view,
# W*cycles-per-window at the root.
#
# Stage 4 is skipped, with a line, on a run dir that has no window artifacts
# (an older run, or one whose filter never got there) -- the run is still fully
# evaluated by stages 2 and 3.
#
# Any extra arguments are forwarded to run_filtering.py as Hydra overrides (and
# used to resolve the run dir), e.g.:
#
#   scripts/run_filtering_pipeline.sh filtering.mode=joint filtering.num_assimilation_windows=4
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

# The ESMDA-schema view (see the header). Rebuilt from scratch each run so a
# rerun never leaves a stale link behind, and `run_summary.yaml` / `eval_fields.nc`
# are deliberately NOT linked: the ESMDA stages write those two names, and a
# symlink would send them straight back into the filtering stages' own files --
# the collision this directory exists to avoid.
ESMDA_VIEW="${RUN_DIR}/esmda_view"
if [ -f "${RUN_DIR}/posterior_state_mean.nc" ] && [ -d "${RUN_DIR}/windows" ]; then
  rm -rf "${ESMDA_VIEW}"
  mkdir -p "${ESMDA_VIEW}"
  (
    cd "${ESMDA_VIEW}"
    ln -s ../windows windows
    for path in ../*.nc ../*.yaml; do
      [ -e "${path}" ] || continue
      name="$(basename "${path}")"
      case "${name}" in
      run_summary.yaml | eval_fields.nc) continue ;;
      esac
      ln -s "../${name}" "${name}"
    done
  )
  pixi run -e dev python scripts/esmda/compute_esmda_metrics.py --run-dir "${ESMDA_VIEW}"
  pixi run -e dev python scripts/esmda/make_esmda_figures.py --run-dir "${ESMDA_VIEW}"
  echo "ESMDA-schema view of this run: ${ESMDA_VIEW}"
else
  echo "No window artifacts in ${RUN_DIR}; skipping the ESMDA-schema stages"
fi

echo "Filtering pipeline complete: ${RUN_DIR}"
