#!/usr/bin/env bash
# Run the full filter-smoothing (hybrid) pipeline: assimilate, then metrics,
# then figures.
#
#   1. scripts/filter_smoothing/run_filter_smoothing.py -- runs the hybrid
#      (ESMDA parameter MDA + sequential filter per window) and saves every
#      artifact.
#   2. scripts/filtering/compute_filtering_metrics.py -- writes run_summary.yaml.
#   3. scripts/filtering/make_filtering_figures.py    -- draws the figures.
#   4. scripts/esmda/compute_esmda_metrics.py + make_esmda_figures.py, run over
#      the WINDOW artifacts the hybrid writes in the ESMDA schema, into
#      <run dir>/esmda_view/ (see "The ESMDA-schema view" below).
#
# The hybrid counterpart of scripts/run_esmda_pipeline.sh and
# scripts/run_filtering_pipeline.sh — structurally a copy of the latter,
# because a hybrid run dir IS a filtering run dir at its root (the window's
# state comes from the filter, cycle-indexed) with the ESMDA window schema
# beside it (the window's parameters come from the MDA loop). All stages share
# one output directory: conf/run_filter_smoothing.yaml's `paths.results_dir`
# (`.temp/filter_smoothing_<truth_model>_to_<assim_model>` by default),
# resolved below from the same config with the same overrides the runner gets,
# so the output location is configured in one place (the YAML), not here.
#
# Stages 2 and 3 reduce over the filter's per-cycle states exactly as they do
# for a pure filtering run, including the `run.ensemble_save_on_disk` trade-off
# documented in run_filtering_pipeline.sh (analyzed frames only vs full
# forecast segments under _ensemble_states/). Two hybrid-specific notes:
#
#   * `filtering.mode=state` runs carry the ESMDA parameters through the filter
#     unmodified, so no params_history.nc is written and the per-cycle
#     parameter panels are skipped (the stages tolerate the absence); the
#     parameter story then lives entirely in the ESMDA-schema view.
#   * In joint mode the root-level parameter artifacts are the FILTER's
#     (corrected) values while the ESMDA-schema view scores the MDA posterior
#     — windows/window_{w}_posterior_params.nc — so the two summaries answer
#     different questions on purpose: "what did the filter run with" vs "what
#     did the smoother estimate".
#
# Because the MDA posterior is the hybrid's parameter estimate, stage 4's
# parameter figures — rollout_time_evolution.png (prior AND posterior parameter
# evolution against the truth), parameter_error.png, parameter_marginals.png —
# are also COPIED from esmda_view/ to the run root, so a hybrid run dir shows
# its parameter story without descending into the view.
#
# The ESMDA-schema view (`<run dir>/esmda_view/`)
# ----------------------------------------------
# Same construction, and same reason, as run_filtering_pipeline.sh's: both
# metric families write `run_summary.yaml` / `eval_fields.nc` and same-named
# figures with DIFFERENT binnings (per window vs per cycle), so they must not
# share a directory. `esmda_view/` holds symlinks to the artifacts both
# families read and is where the ESMDA stages write their own outputs; nothing
# is copied or renamed. D3 (the observation-space data mismatch) reads
# windows/window_{w}_pred_obs.nc — the FILTER phase's per-cycle prior/posterior
# rows, as in a filtering run. The MDA loop's own observation-space record is
# windows/window_{w}_esmda_pred_obs.nc, which no shared stage reads (its
# esmda_step axis has NO posterior entry — see the file's attrs); it is there
# for bespoke analysis, not for D3.
#
# Stage 4 is skipped, with a line, on a run dir that has no window artifacts.
#
# Any extra arguments are forwarded to run_filter_smoothing.py as Hydra
# overrides (and used to resolve the run dir), e.g.:
#
#   scripts/run_filter_smoothing_pipeline.sh filtering.mode=joint \
#       esmda/smoother=dynamic params@prior_params=dynamic
#   scripts/run_filter_smoothing_pipeline.sh filtering.mode=state \
#       filter_smoothing.num_assimilation_windows=4
#   scripts/run_filter_smoothing_pipeline.sh esmda/smoother=static \
#       params@prior_params=static params@truth_params=static_truth
#   scripts/run_filter_smoothing_pipeline.sh model@truth_model=pylbm \
#       model@assim_model=pylbm run.ensemble_save_on_disk=true
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
echo "Filter-smoothing pipeline output dir: ${RUN_DIR}"

pixi run -e cuda python scripts/filter_smoothing/run_filter_smoothing.py "$@"
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

  # Surface the ESMDA-schema PARAMETER figures at the run root. The hybrid's
  # parameter story is the MDA posterior, and these are its prior/posterior
  # panels (rollout_time_evolution.png draws the prior AND posterior parameter
  # evolution against the truth); the root-level filtering figures only cover
  # the filter's own params_history, which mode=state does not even write.
  # Copied rather than symlinked so the root stays self-contained; the names
  # collide with nothing the filtering stages write.
  for fig in rollout_time_evolution.png parameter_error.png parameter_marginals.png; do
    if [ -f "${ESMDA_VIEW}/${fig}" ]; then
      cp "${ESMDA_VIEW}/${fig}" "${RUN_DIR}/${fig}"
    fi
  done
else
  echo "No window artifacts in ${RUN_DIR}; skipping the ESMDA-schema stages"
fi

echo "Filter-smoothing pipeline complete: ${RUN_DIR}"
