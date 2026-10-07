#!/usr/bin/env bash
# Queue a campaign chain behind whatever is running now: block until the named
# chain(s) and every per-method campaign have released their locks, then exec
# run_all_campaigns.sh with the environment it was given.
#
# Exists so a chain can be launched NOW and start LATER, unattended. Launching
# run_all_campaigns.sh directly while another chain is mid-flight does not queue
# — the second chain's first campaign hits a held `driver.lock` and dies.
#
# WAIT_FOR names the chain dirs to wait on (space separated, default: every
# `_chain*` dir that exists under RESULTS_ROOT). The per-method campaign locks
# are always waited on as well, so this also queues behind a bare campaign
# started by hand.
#
# Everything else (STEPS, CHAIN_NAME, the axis knobs) is passed straight through
# to run_all_campaigns.sh.
#
# Usage:
#   STEPS="pyudales esmda|pypalm esmda" CHAIN_NAME=_chain_next \
#   OBS_INTERVAL_LIST=15.0 \
#     setsid nohup bash job_scripts/local/experiments/wait_then_run.sh \
#       > /dev/null 2>&1 < /dev/null &
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}/../../.."

# shellcheck source=job_scripts/local/experiments/settings.sh
source "${HERE}/settings.sh"
[[ "${RESULTS_ROOT}" = /* ]] || RESULTS_ROOT="$(pwd)/${RESULTS_ROOT}"

QUEUE_LOG="${RESULTS_ROOT}/${CHAIN_NAME:-_chain}.queue.log"
mkdir -p "$(dirname "${QUEUE_LOG}")"

note () { echo "$(date -Is) $*" >> "${QUEUE_LOG}"; }

# Lock files to wait on: the named (or discovered) chains, plus every campaign.
locks=()
if [[ -n "${WAIT_FOR:-}" ]]; then
  for name in ${WAIT_FOR}; do locks+=("${RESULTS_ROOT}/${name}/chain.lock"); done
else
  shopt -s nullglob
  locks+=("${RESULTS_ROOT}"/_chain*/chain.lock)
  shopt -u nullglob
fi
for method in esmda filtering filter_smoothing; do
  locks+=("${RESULTS_ROOT}/${method}/_logs/driver.lock")
done

note "QUEUED: waiting on ${#locks[@]} lock(s) before starting ${CHAIN_NAME:-_chain}"
for lock in "${locks[@]}"; do
  [[ -f "${lock}" ]] || continue
  # Skip the lock this chain will take itself.
  [[ "${lock}" == "${RESULTS_ROOT}/${CHAIN_NAME:-_chain}/chain.lock" ]] && continue
  if ! flock -n "${lock}" true 2>/dev/null; then
    note "  waiting for ${lock}"
    flock "${lock}" true          # blocks until the holder releases it
    note "  released: ${lock}"
  fi
done

# A campaign lock is released between campaigns of a running chain, so re-check
# that every lock is free at the same instant before starting; otherwise this
# could slip into the gap between two campaigns of the chain ahead.
while :; do
  busy=""
  for lock in "${locks[@]}"; do
    [[ -f "${lock}" ]] || continue
    [[ "${lock}" == "${RESULTS_ROOT}/${CHAIN_NAME:-_chain}/chain.lock" ]] && continue
    flock -n "${lock}" true 2>/dev/null || busy="${lock}"
  done
  [[ -z "${busy}" ]] && break
  note "  still busy (${busy}); re-checking in 60 s"
  sleep 60
done

note "CLEAR: starting ${CHAIN_NAME:-_chain}"
exec bash "${HERE}/${CHAIN_SCRIPT:-run_all_campaigns.sh}"
