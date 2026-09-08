#!/usr/bin/env bash
# Filter-smoothing (hybrid) campaign: scripts/run_filter_smoothing_pipeline.sh
# over the shared experiment axes.
#
# One run per point of
#   TRUTH_MODEL_LIST x NUM_WINDOWS_LIST x LOCALIZATION_LIST x OBS_INTERVAL_LIST x INFLOW_LIST
# with everything else pinned by job_scripts/local/experiments/settings.sh.
# Each run goes through the full pipeline: run_filter_smoothing.py, the
# filtering metric/figure stages at the run root, and the ESMDA-schema view
# (whose parameter figures are copied back to the root).
#
# Axis -> override, for this entry point:
#   truth model       model@truth_model (pyudales or pypalm; the assimilation
#                     mount stays ASSIM_MODEL, so a pypalm truth is a
#                     cross-model run) plus that mount's forcing knobs
#   num windows       filter_smoothing.num_assimilation_windows
#   localization      esmda/localization AND/OR filtering/localization — the
#                     hybrid has two updates (the MDA parameter loop and the
#                     filter's state analysis) and therefore two localization
#                     mounts. LOCALIZATION_SCOPE picks which the axis drives:
#                       both     (default) — one setting for the whole run
#                       filter   — the filter's state analysis only; the MDA
#                                  loop stays unlocalized (conf's default)
#                       smoother — the MDA parameter loop only
#   obs interval      esmda.interval_seconds — SMOOTHER-SIDE ONLY. The filter
#                     half still assimilates one frame per cycle; this bins the
#                     observations the MDA parameter loop sees.
#   inflow            both model mounts' boundary_condition / inlet_turbulence
#
# Method knobs, defaulted here and env-overridable:
#   ESMDA_SMOOTHER   static|dynamic — the state-bearing smoothers are rejected
#                    by the hybrid's constructor (the filter owns the state)
#   PRIOR_PARAMS     static|dynamic — must match the smoother
#   FILTERING_MODE   state|joint (`parameter` is not a hybrid mode)
#   NUM_ESMDA_STEPS  esmda.num_steps
#   ASSIMILATE_EVERY_N_STEP  the filter's analysis stride. Under a stride the
#                    thinning applies to BOTH halves, so keep the observation
#                    interval coarse enough that every bin holds a strided frame.
#
# Usage:
#   bash job_scripts/local/experiments/run_filter_smoothing_experiments.sh
#   DRY_RUN=1 bash job_scripts/local/experiments/run_filter_smoothing_experiments.sh
#   FILTERING_MODE=state LOCALIZATION_SCOPE=filter \
#     bash job_scripts/local/experiments/run_filter_smoothing_experiments.sh
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
: "${FILTERING_MODE:=state}"
: "${FILTERING_ANALYSIS:=stochastic}"
: "${FILTERING_INFLATION:=rtps}"
: "${FILTERING_EVOLUTION:=none}"
: "${FILTERING_STATE_REDUCTION:=none}"
: "${ASSIMILATE_EVERY_N_STEP:=1}"
: "${LOCALIZATION_SCOPE:=both}"
: "${ENSEMBLE_SAVE_ON_DISK:=false}"

case "${LOCALIZATION_SCOPE}" in
both | filter | smoother) ;;
*) fatal "LOCALIZATION_SCOPE must be both|filter|smoother, got '${LOCALIZATION_SCOPE}'" ;;
esac

campaign_init filter_smoothing
check_cycle_tiling "${ASSIMILATE_EVERY_N_STEP}"

for truth in ${TRUTH_MODEL_LIST}; do
  for inflow in ${INFLOW_LIST}; do
    mapfile -t model_args < <(model_overrides "${truth}" "${inflow}")
    ((${#model_args[@]} > 1)) || fatal "no overrides for TRUTH=${truth} INFLOW=${inflow}"
    for windows in ${NUM_WINDOWS_LIST}; do
      for loc in ${LOCALIZATION_LIST}; do
        # Same analysis/localization compatibility rule as the pure filter — it
        # is the same filter object, built from the same groups.
        case "${FILTERING_ANALYSIS}" in
        etkf | etkf_tsvd)
          [[ "${loc}" == none || "${LOCALIZATION_SCOPE}" == smoother ]] || {
            echo "SKIP localization=${loc}: analysis=${FILTERING_ANALYSIS} requires none"
            continue
          }
          ;;
        letkf | letkf_tsvd)
          [[ "${loc}" != none || "${LOCALIZATION_SCOPE}" == smoother ]] || {
            echo "SKIP localization=none: analysis=${FILTERING_ANALYSIS} requires a localization"
            continue
          }
          ;;
        esac

        case "${LOCALIZATION_SCOPE}" in
        both) loc_args=("esmda/localization=${loc}" "filtering/localization=${loc}") ;;
        filter) loc_args=("esmda/localization=none" "filtering/localization=${loc}") ;;
        smoother) loc_args=("esmda/localization=${loc}" "filtering/localization=none") ;;
        esac

        for interval in ${OBS_INTERVAL_LIST}; do
          id="$(run_id "${truth}_to_${ASSIM_MODEL}" "w${windows}" "loc${loc}" \
            "obs$(tag "${interval}")" "${inflow}")"
          # shellcheck disable=SC2086
          enqueue "${id}" \
            "${COMMON_ARGS[@]}" \
            "${model_args[@]}" \
            "${loc_args[@]}" \
            "esmda/smoother=${ESMDA_SMOOTHER}" \
            "params@prior_params=${PRIOR_PARAMS}" \
            "esmda.interval_seconds=${interval}" \
            "esmda.num_steps=${NUM_ESMDA_STEPS}" \
            "filtering.mode=${FILTERING_MODE}" \
            "filtering/analysis=${FILTERING_ANALYSIS}" \
            "filtering/inflation=${FILTERING_INFLATION}" \
            "filtering/evolution=${FILTERING_EVOLUTION}" \
            "filtering/state_reduction=${FILTERING_STATE_REDUCTION}" \
            "filtering.assimilate_every_n_step=${ASSIMILATE_EVERY_N_STEP}" \
            "filter_smoothing.num_assimilation_windows=${windows}" \
            "filter_smoothing.seed=${SEED}" \
            "filter_smoothing.obs_error_std=${OBS_ERROR_STD}" \
            "run.ensemble_save_on_disk=${ENSEMBLE_SAVE_ON_DISK}" \
            ${EXTRA_ARGS}
        done
      done
    done
  done
done

drain scripts/run_filter_smoothing_pipeline.sh
