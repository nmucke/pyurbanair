#!/usr/bin/env bash
# State-only filtering: three spread-control variations, run back to back.
#
# The filter estimates the STATE alone (`filtering.mode=state`) — the sampled
# parameters ride through every cycle unmodified, so this isolates the state
# update from the parameter estimation the joint runs do.
#
#   1. noloc_noinfl   no localization, no inflation
#   2. noloc_infl     no localization, RTPS inflation
#   3. loc_infl       correlation localization, RTPS inflation
#
# `filtering/evolution=none` in all three: EnsembleKalmanFilter REJECTS a
# parameter evolution in state mode (it would have no effect — parameters are
# carried unmodified). The "no inflation" variation is legal precisely because
# the spread-maintenance guard only binds in parameter/joint mode, where an
# un-inflated parameter block collapses; a state ensemble regenerates its own
# spread through the forecast.
#
# Each variation is a separate campaign over the INFLOW axis, tagged into its
# run ids (RUN_ID_TAG) so the three land side by side in the same filtering/
# results tree without colliding with each other or with the joint runs.
# Localization is already in the id, so the tag carries the inflation alone.
#
# pyudales -> pyudales only.
#
# Usage (queued behind whatever is running):
#   CHAIN_SCRIPT=run_state_only_variations.sh CHAIN_NAME=_chain_state_only \
#     setsid nohup bash job_scripts/local/experiments/wait_then_run.sh \
#       > /dev/null 2>&1 < /dev/null &
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}/../../.."

# shellcheck source=job_scripts/local/experiments/settings.sh
source "${HERE}/settings.sh"
[[ "${RESULTS_ROOT}" = /* ]] || RESULTS_ROOT="$(pwd)/${RESULTS_ROOT}"

CHAIN_DIR="${RESULTS_ROOT}/${CHAIN_NAME:-_chain_state_only}"
mkdir -p "${CHAIN_DIR}"
CHAIN_LOG="${CHAIN_DIR}/chain.log"
CHAIN_TSV="${CHAIN_DIR}/chain_progress.tsv"

exec 9> "${CHAIN_DIR}/chain.lock"
if ! flock -n 9; then
  echo "FATAL: another chain holds ${CHAIN_DIR}/chain.lock" >&2
  exit 1
fi
[[ -f "${CHAIN_TSV}" ]] ||
  printf 'step\tvariation\tlocalization\tinflation\trc\tstarted\tended\telapsed_s\n' > "${CHAIN_TSV}"

# "<name> <localization> <inflation>"
VARIATIONS=(
  "noloc_noinfl none none"
  "noloc_infl none rtps"
  "loc_infl correlation rtps"
)

{
  echo "$(date -Is) STATE-ONLY CHAIN START: ${#VARIATIONS[@]} variations -> ${RESULTS_ROOT}/filtering"
  df -h "${RESULTS_ROOT}" | tail -1
} >> "${CHAIN_LOG}"

step=0
for entry in "${VARIATIONS[@]}"; do
  step=$((step + 1))
  read -r name loc infl <<< "${entry}"
  log="${CHAIN_DIR}/${step}_${name}.log"
  started="$(date -Is)"
  t0="$(date +%s)"
  echo "$(date -Is) [${step}/${#VARIATIONS[@]}] START ${name} (localization=${loc}, inflation=${infl})" \
    >> "${CHAIN_LOG}"

  TRUTH_MODEL_LIST="pyudales" \
    FILTERING_MODE="state" \
    FILTERING_EVOLUTION="none" \
    FILTERING_INFLATION="${infl}" \
    LOCALIZATION_LIST="${loc}" \
    RUN_ID_TAG="state_infl${infl}" \
    bash "${HERE}/run_filtering_experiments.sh" > "${log}" 2>&1
  rc=$?

  t1="$(date +%s)"
  printf '%d\t%s\t%s\t%s\t%d\t%s\t%s\t%d\n' \
    "${step}" "${name}" "${loc}" "${infl}" "${rc}" "${started}" "$(date -Is)" "$((t1 - t0))" \
    >> "${CHAIN_TSV}"
  echo "$(date -Is) [${step}/${#VARIATIONS[@]}] END ${name} rc=${rc} elapsed=$((t1 - t0))s" >> "${CHAIN_LOG}"
  df -h "${RESULTS_ROOT}" | tail -1 >> "${CHAIN_LOG}"
done

echo "$(date -Is) STATE-ONLY CHAIN DONE" >> "${CHAIN_LOG}"
column -t -s $'\t' "${CHAIN_TSV}" >> "${CHAIN_LOG}" 2>/dev/null
