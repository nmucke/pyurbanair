#!/usr/bin/env bash
# Filtering (EnKF) campaign: scripts/run_filtering_pipeline.sh over the shared
# experiment axes.
#
# One run per point of
#   TRUTH_MODEL_LIST x NUM_WINDOWS_LIST x LOCALIZATION_LIST x INFLOW_LIST
# with everything else pinned by job_scripts/local/experiments/settings.sh.
# Each run goes through the full pipeline: run_filtering.py, the filtering
# metric/figure stages, and the ESMDA-schema view over the window artifacts.
#
# OBS_INTERVAL_LIST DOES NOT APPLY HERE. The filter assimilates individual
# frames — one analysis cycle per observation interval (`OUTPUT_FREQUENCY`) —
# and never aggregates; observation binning is a smoother knob. The comparable
# filter axis is the analysis STRIDE, `filtering.assimilate_every_n_step`
# (below), which thins the analyses without touching the output cadence. It is a
# constant here, not an axis, so the ESMDA and filtering campaigns keep the same
# run count per (windows x localization x inflow) cell.
#
# Axis -> override, for this entry point:
#   truth model       model@truth_model (pyudales or pypalm; the assimilation
#                     mount stays ASSIM_MODEL, so a pypalm truth is a
#                     cross-model run) plus that mount's forcing knobs
#   num windows       filtering.num_assimilation_windows
#   localization      filtering/localization=correlation|none
#   inflow            both model mounts' boundary_condition / inlet_turbulence
#
# Method knobs, defaulted here and env-overridable:
#   FILTERING_MODE     state|parameter|joint
#   FILTERING_ANALYSIS stochastic|etkf|etkf_tsvd|letkf|letkf_tsvd
#   FILTERING_INFLATION  state spread maintenance (rtps)
#   FILTERING_EVOLUTION  parameter forecast: random_walk (default) or none
#   ASSIMILATE_EVERY_N_STEP  analysis stride (see above)
#   PRIOR_PARAMS       MUST be static: the filter supports static parameters
#                      only (a dynamic prior belongs to the smoothers), while
#                      TRUTH_PARAMS stays the shared dynamic_sine so the filter
#                      is scored on tracking a drifting truth.
#
# Usage:
#   bash job_scripts/local/experiments/run_filtering_experiments.sh
#   DRY_RUN=1 bash job_scripts/local/experiments/run_filtering_experiments.sh
#   FILTERING_MODE=state bash job_scripts/local/experiments/run_filtering_experiments.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}/../../.."

# shellcheck source=job_scripts/local/experiments/settings.sh
source "${HERE}/settings.sh"
# shellcheck source=job_scripts/local/experiments/campaign_lib.sh
source "${HERE}/campaign_lib.sh"

: "${FILTERING_MODE:=joint}"
: "${FILTERING_ANALYSIS:=stochastic}"
: "${FILTERING_INFLATION:=rtps}"
# Additive Gaussian random walk on the parameters between cycles — the standard
# augmented-state parameter forecast, and what lets a STATIC prior track the
# drifting (dynamic_sine) truth instead of only shrinking around its first
# estimate. `std` is one scalar for every parameter; set it per parameter with
#   EXTRA_ARGS="filtering.parameter_evolution.std={inflow_angle:1.0,velocity_magnitude:0.05}"
# It runs alongside the inflation above, which maintains STATE spread.
: "${FILTERING_EVOLUTION:=random_walk}"
: "${FILTERING_STATE_REDUCTION:=none}"
: "${ASSIMILATE_EVERY_N_STEP:=1}"
: "${PRIOR_PARAMS:=static}"
# The filter keeps only the per-cycle analyzed frames by default. Set true for
# within-cycle turbulence statistics (every member's full forecast segment under
# _ensemble_states/) at ensemble x cycles x segment of disk.
: "${ENSEMBLE_SAVE_ON_DISK:=false}"
# Appended to every run id in this campaign. The id encodes only the swept axes,
# so two campaigns differing in a METHOD knob (mode, inflation, evolution) would
# otherwise write into the same run dirs — and the second would be skipped as
# already done. Tag them apart, e.g. RUN_ID_TAG=state_inflnone.
: "${RUN_ID_TAG:=}"

# Delete state_history.nc once the whole pipeline (assimilation, metrics,
# figures, ESMDA-schema view) has finished with it. It duplicates
# windows/window_*_posterior_state.nc frame for frame — ~7 GB per run at one
# analysis per output frame — and the window files are the ones the
# ESMDA-schema stages cannot do without. See prune_state_history in
# campaign_lib.sh, which refuses to delete unless those window files are there.
# The cost is on RE-analysis only: re-running the filtering-native stages
# afterwards falls back to posterior_state.nc (the final frame alone), so their
# per-cycle state RMSE covers the last cycle only. Everything the pipeline
# already wrote is unaffected. Set false to keep the file.
: "${PRUNE_STATE_HISTORY:=true}"
if [[ "${PRUNE_STATE_HISTORY}" == true ]]; then
  POST_RUN_HOOK=prune_state_history
fi

campaign_init filtering
check_cycle_tiling "${ASSIMILATE_EVERY_N_STEP}"

for truth in ${TRUTH_MODEL_LIST}; do
  for inflow in ${INFLOW_LIST}; do
    mapfile -t model_args < <(model_overrides "${truth}" "${inflow}")
    ((${#model_args[@]} > 1)) || fatal "no overrides for TRUTH=${truth} INFLOW=${inflow}"
    for windows in ${NUM_WINDOWS_LIST}; do
      for loc in ${LOCALIZATION_LIST}; do
        # The analysis and the localization are not independent: the ETKF
        # variants are formulated without localization, the LETKF variants
        # require one. Skip rather than let the run die in the constructor.
        case "${FILTERING_ANALYSIS}" in
        etkf | etkf_tsvd)
          [[ "${loc}" == none ]] || {
            echo "SKIP localization=${loc}: analysis=${FILTERING_ANALYSIS} requires none"
            continue
          }
          ;;
        letkf | letkf_tsvd)
          [[ "${loc}" != none ]] || {
            echo "SKIP localization=none: analysis=${FILTERING_ANALYSIS} requires a localization"
            continue
          }
          ;;
        esac

        id_parts=("${truth}_to_${ASSIM_MODEL}" "w${windows}" "loc${loc}" "${inflow}")
        [[ -n "${RUN_ID_TAG}" ]] && id_parts+=("${RUN_ID_TAG}")
        id="$(run_id "${id_parts[@]}")"
        # shellcheck disable=SC2086
        enqueue "${id}" \
          "${COMMON_ARGS[@]}" \
          "${model_args[@]}" \
          "params@prior_params=${PRIOR_PARAMS}" \
          "filtering.mode=${FILTERING_MODE}" \
          "filtering/analysis=${FILTERING_ANALYSIS}" \
          "filtering/localization=${loc}" \
          "filtering/inflation=${FILTERING_INFLATION}" \
          "filtering/evolution=${FILTERING_EVOLUTION}" \
          "filtering/state_reduction=${FILTERING_STATE_REDUCTION}" \
          "filtering.num_assimilation_windows=${windows}" \
          "filtering.assimilate_every_n_step=${ASSIMILATE_EVERY_N_STEP}" \
          "filtering.seed=${SEED}" \
          "filtering.obs_error_std=${OBS_ERROR_STD}" \
          "run.ensemble_save_on_disk=${ENSEMBLE_SAVE_ON_DISK}" \
          ${EXTRA_ARGS}
      done
    done
  done
done

drain scripts/run_filtering_pipeline.sh
