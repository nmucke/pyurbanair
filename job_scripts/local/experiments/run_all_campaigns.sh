#!/usr/bin/env bash
# Run every DA-method campaign, one after the next, in one detached process.
#
# Order (each starts only when the previous has fully drained — assimilation,
# metrics, figures and the ESMDA-schema view of its last run included):
#
#   1. pyudales -> pyudales   esmda
#   2. pyudales -> pyudales   filtering
#   3. pyudales -> pyudales   filter_smoothing
#   4. pypalm   -> pyudales   esmda
#   5. pypalm   -> pyudales   filtering
#   6. pypalm   -> pyudales   filter_smoothing
#
# Sequential on purpose: NUM_LANES is per campaign, and two campaigns at once
# would put 2 x WORKERS members on a box that is DRAM-bandwidth-bound past ~4-8.
#
# A campaign that fails does NOT abort the chain — its exit code is recorded and
# the next one starts. Losing five campaigns because the first hit a bad member
# is the worst outcome; each campaign is independently resumable anyway (its own
# `_logs/<id>.ok` markers), so a failed one can be re-run afterwards and will
# skip whatever finished.
#
# Usage (detached, survives logout):
#   setsid nohup bash job_scripts/local/experiments/run_all_campaigns.sh \
#     > /dev/null 2>&1 < /dev/null &
#
# Watch:
#   tail -f <RESULTS_ROOT>/_chain/chain.log         # which campaign is running
#   column -t -s $'\t' <RESULTS_ROOT>/<method>/_logs/progress.tsv   # per-run results
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}/../../.."

# Only for RESULTS_ROOT and the shared knobs; each campaign re-sources it.
# shellcheck source=job_scripts/local/experiments/settings.sh
source "${HERE}/settings.sh"

[[ "${RESULTS_ROOT}" = /* ]] || RESULTS_ROOT="$(pwd)/${RESULTS_ROOT}"
# A chain gets its own log/lock/progress dir, so a second chain neither waits on
# the first one's lock nor appends its steps to the first one's progress file.
CHAIN_DIR="${RESULTS_ROOT}/${CHAIN_NAME:-_chain}"
mkdir -p "${CHAIN_DIR}"
CHAIN_LOG="${CHAIN_DIR}/chain.log"
CHAIN_TSV="${CHAIN_DIR}/chain_progress.tsv"

# One chain at a time.
exec 9> "${CHAIN_DIR}/chain.lock"
if ! flock -n 9; then
  echo "FATAL: another chain holds ${CHAIN_DIR}/chain.lock" >&2
  exit 1
fi

[[ -f "${CHAIN_TSV}" ]] ||
  printf 'step\ttruth\tmethod\trc\tstarted\tended\telapsed_s\n' > "${CHAIN_TSV}"

# "<truth model> <method>", in run order. Override with STEPS to chain a
# different set, e.g. an axis sweep over two methods and both truth models:
#   STEPS="pyudales esmda|pyudales filter_smoothing|pypalm esmda|pypalm filter_smoothing"
# Axis knobs (OBS_INTERVAL_LIST, LOCALIZATION_LIST, INFLOW_LIST, ...) are read
# from the environment by each campaign, so export them alongside STEPS.
if [[ -n "${STEPS:-}" ]]; then
  IFS='|' read -r -a STEPS <<< "${STEPS}"
else
  STEPS=(
    "pyudales esmda"
    "pyudales filtering"
    "pyudales filter_smoothing"
    "pypalm esmda"
    "pypalm filtering"
    "pypalm filter_smoothing"
  )
fi

{
  echo "$(date -Is) CHAIN START: ${#STEPS[@]} campaigns -> ${RESULTS_ROOT}"
  df -h "${RESULTS_ROOT}" | tail -1
} >> "${CHAIN_LOG}"

step=0
for entry in "${STEPS[@]}"; do
  step=$((step + 1))
  read -r truth method <<< "${entry}"
  log="${CHAIN_DIR}/${step}_${truth}_${method}.log"
  started="$(date -Is)"
  t0="$(date +%s)"
  echo "$(date -Is) [${step}/${#STEPS[@]}] START ${truth} -> ${ASSIM_MODEL}, ${method} (log: ${log})" \
    >> "${CHAIN_LOG}"

  TRUTH_MODEL_LIST="${truth}" \
    bash "${HERE}/run_${method}_experiments.sh" > "${log}" 2>&1
  rc=$?

  t1="$(date +%s)"
  printf '%d\t%s\t%s\t%d\t%s\t%s\t%d\n' \
    "${step}" "${truth}" "${method}" "${rc}" "${started}" "$(date -Is)" "$((t1 - t0))" \
    >> "${CHAIN_TSV}"
  echo "$(date -Is) [${step}/${#STEPS[@]}] END ${truth} ${method} rc=${rc} elapsed=$((t1 - t0))s" \
    >> "${CHAIN_LOG}"
  df -h "${RESULTS_ROOT}" | tail -1 >> "${CHAIN_LOG}"
done

echo "$(date -Is) CHAIN DONE" >> "${CHAIN_LOG}"
column -t -s $'\t' "${CHAIN_TSV}" >> "${CHAIN_LOG}" 2>/dev/null
