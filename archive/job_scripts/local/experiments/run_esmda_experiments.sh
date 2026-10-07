#!/usr/bin/env bash
# ESMDA campaign: scripts/run_esmda_pipeline.sh over the shared experiment axes.
#
# One run per point of
#   TRUTH_MODEL_LIST x NUM_WINDOWS_LIST x LOCALIZATION_LIST x OBS_INTERVAL_LIST x INFLOW_LIST
# with everything else pinned by job_scripts/local/experiments/settings.sh —
# edit that file (or export the same names) to retune all three campaigns at
# once. Each run goes through the full pipeline: run_esmda.py, then
# compute_esmda_metrics.py, then make_esmda_figures.py.
#
# Axis -> override, for this entry point:
#   truth model       model@truth_model (pyudales or pypalm; the assimilation
#                     mount stays ASSIM_MODEL, so a pypalm truth is a
#                     cross-model run) plus that mount's forcing knobs
#   num windows       esmda.num_assimilation_windows
#   localization      esmda/localization=correlation|none
#   obs interval      esmda.interval_seconds   (window observations are binned
#                                               this wide and averaged)
#   inflow            both model mounts' boundary_condition / inlet_turbulence
#
# Method knobs, defaulted here and env-overridable (they are ESMDA-specific, so
# they live here rather than in settings.sh):
#   ESMDA_SMOOTHER   static|state_and_parameter|dynamic|state_and_dynamic
#   PRIOR_PARAMS     static|dynamic — must match the smoother
#   NUM_ESMDA_STEPS  esmda.num_steps (MDA iterations per window)
#
# Usage:
#   bash job_scripts/local/experiments/run_esmda_experiments.sh
#   DRY_RUN=1 bash job_scripts/local/experiments/run_esmda_experiments.sh
#   ESMDA_SMOOTHER=static PRIOR_PARAMS=static TRUTH_PARAMS=static_truth \
#     bash job_scripts/local/experiments/run_esmda_experiments.sh
#
# Detached (the campaign outlives the shell):
#   setsid nohup bash job_scripts/local/experiments/run_esmda_experiments.sh \
#     > /dev/null 2>&1 < /dev/null &
#   # progress: ${RESULTS_ROOT}/esmda/_logs/{driver,<id>}.log, progress.tsv
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}/../../.."

# shellcheck source=job_scripts/local/experiments/settings.sh
source "${HERE}/settings.sh"
# shellcheck source=job_scripts/local/experiments/campaign_lib.sh
source "${HERE}/campaign_lib.sh"

: "${ESMDA_SMOOTHER:=dynamic}"
: "${PRIOR_PARAMS:=dynamic}"
: "${NUM_ESMDA_STEPS:=3}"
# The smoothers save every member's forecast to disk rather than holding one
# in-memory ensemble Dataset (the entry point's own default); the prior state
# ensemble is the large artifact and is off, as it is in conf/run_esmda.yaml.
: "${ENSEMBLE_SAVE_ON_DISK:=true}"
: "${SAVE_PRIOR_STATE:=false}"

campaign_init esmda

for truth in ${TRUTH_MODEL_LIST}; do
  for inflow in ${INFLOW_LIST}; do
    mapfile -t model_args < <(model_overrides "${truth}" "${inflow}")
    ((${#model_args[@]} > 1)) || fatal "no overrides for TRUTH=${truth} INFLOW=${inflow}"
    for windows in ${NUM_WINDOWS_LIST}; do
      for loc in ${LOCALIZATION_LIST}; do
        for interval in ${OBS_INTERVAL_LIST}; do
          id="$(run_id "${truth}_to_${ASSIM_MODEL}" "w${windows}" "loc${loc}" \
            "obs$(tag "${interval}")" "${inflow}")"
          # shellcheck disable=SC2086
          enqueue "${id}" \
            "${COMMON_ARGS[@]}" \
            "${model_args[@]}" \
            "esmda/smoother=${ESMDA_SMOOTHER}" \
            "params@prior_params=${PRIOR_PARAMS}" \
            "esmda/localization=${loc}" \
            "esmda.num_assimilation_windows=${windows}" \
            "esmda.interval_seconds=${interval}" \
            "esmda.num_steps=${NUM_ESMDA_STEPS}" \
            "esmda.seed=${SEED}" \
            "esmda.obs_error_std=${OBS_ERROR_STD}" \
            "run.ensemble_save_on_disk=${ENSEMBLE_SAVE_ON_DISK}" \
            "run.save_prior_state=${SAVE_PRIOR_STATE}" \
            ${EXTRA_ARGS}
        done
      done
    done
  done
done

drain scripts/run_esmda_pipeline.sh
