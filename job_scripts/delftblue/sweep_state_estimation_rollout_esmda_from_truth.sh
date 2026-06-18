#!/bin/bash
# DELFTBLUE sweep over the STATE-ESTIMATION METHODS for rollout-ESMDA-from-truth.
#
# Unlike the domain/ensemble/steps/interval sweeps (sweep_base.sh), which vary
# one numeric knob, this sweep submits the SAME experiment once per state-update
# strategy of the joint state + time-varying-parameter smoother
# (esmda/smoother=state_and_dynamic). One sbatch per method, reusing the
# per-backend rollout_esmda_from_truth.slurm runners:
#
#   corr_ic   correlation-based localization (Vossepoel et al. 2025), update
#             applied to the window initial condition
#   dist_ic   physical-distance-based localization, update applied to the
#             window initial condition
#   svd_ic    online reduced-SVD state update (esmda/state_reduction=svd),
#             basis fitted to ALL window snapshots, update applied to the
#             window initial condition only
#   svd_all   as svd_ic, plus the post-loop joint Kalman update of the state
#             at EVERY window time step (esmda.final_time_smoothing=true;
#             needs the in-memory ensemble, i.e. run.results_dir unset)
#
# Experiment shape (chosen to match the loaded ground truth, which carries a
# parameter knot every ~30 s):
#   * 30 s assimilation windows with 2 parameter knots per window (the window
#     start and end times)
#   * 10 s observation aggregation intervals (3 obs bins per window)
#   * 13 windows -> 390 s total assimilated horizon (~400 s compute budget)
#
# Usage (from the repo root; extra args are forwarded as Hydra overrides to
# EVERY job):
#
#     bash job_scripts/delftblue/sweep_state_estimation_rollout_esmda_from_truth.sh <pylbm|pyudales|pypalm> [overrides...]
#     for m in pyudales pylbm pypalm; do
#       bash job_scripts/delftblue/sweep_state_estimation_rollout_esmda_from_truth.sh $m
#     done
#
# Knobs (env vars, with defaults):
#   SIMULATION_TIME      window length [s]                  (30.0)
#   NUM_TIME_POINTS      parameter knots per window         (2)
#   NUM_ASSIM_WINDOWS    windows (x window length = total)  (13)
#   INTERVAL_SECONDS     obs aggregation bin width [s]      (10.0)
#   NX/NY/NZ             grid                               (60/80/16)
#   ENSEMBLE_SIZE        ensemble members (= cores, <=64)   (64)
#   NUM_ESMDA_STEPS      ESMDA iterations                   (3)
#   WALLTIME             per-job wall clock                 (12:00:00)
#   SVD_ENERGY           reduced-SVD retained-energy frac.  (0.99)
#   METHODS              space-separated subset to submit   (all four)
#
# Each job lands in its own RESULTS_DIR: the runner's RUN_TAG gets the method
# as RUN_SUFFIX (plus the localization tag for the localized methods), so e.g.
# .../pyudales_nx60_ny80_nz16_ens64_steps3_int10.0_localization_corr_ic.
set -uo pipefail

MODEL="${1:?usage: sweep_state_estimation_rollout_esmda_from_truth.sh <pylbm|pyudales|pypalm> [hydra overrides...]}"
shift
case "${MODEL}" in
  pylbm|pyudales|pypalm) ;;
  *) echo "error: unknown model '${MODEL}' (expected pylbm|pyudales|pypalm)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/${MODEL}/rollout_esmda_from_truth.slurm"
[ -f "${RUNNER}" ] || { echo "error: runner '${RUNNER}' not found" >&2; exit 1; }

# ============================================================================
# Experiment shape (shared by all four methods; see header).
# ============================================================================
SIMULATION_TIME="${SIMULATION_TIME:-30.0}"
# SIMULATION_TIME="${SIMULATION_TIME:-15.0}"
NUM_TIME_POINTS="${NUM_TIME_POINTS:-2}"
NUM_ASSIM_WINDOWS="${NUM_ASSIM_WINDOWS:-10}"
# NUM_ASSIM_WINDOWS="${NUM_ASSIM_WINDOWS:-1}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-6.0}"
NX="${NX:-60}"
NY="${NY:-80}"
NZ="${NZ:-16}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-128}"
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"
# NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-1}"
# WALLTIME="${WALLTIME:-22:00:00}"
WALLTIME="${WALLTIME:-23:00:00}"
SVD_ENERGY="${SVD_ENERGY:-0.99}"
METHODS="${METHODS:-corr_ic dist_ic svd_ic svd_all}"
# ============================================================================

# One core per ensemble member, capped at one 64-core compute-p2 node;
# compute-p1 (48-core nodes, far more of them) when the request fits.
CORES_REQ=${ENSEMBLE_SIZE}
if (( CORES_REQ > 64 )); then
  echo "warning: ensemble_size=${ENSEMBLE_SIZE} exceeds one compute-p2 node (64 cores);" >&2
  echo "         requesting 64 cores; the run still uses num_parallel=${ENSEMBLE_SIZE} (oversubscribed)." >&2
  CORES_REQ=64
fi
if (( CORES_REQ <= 48 )); then PARTITION="compute-p1"; else PARTITION="compute-p2"; fi

# Hydra overrides forwarded to every job.
FORWARD=( "$@" )

# Submit one method. Args: method name, USE_LOCALIZATION value, then any
# method-specific Hydra overrides. The shared experiment shape and the method
# selection travel via --export (the runner + common.sh read them from the
# environment); the state-reduction settings are ordinary Hydra overrides.
submit_method() {
  local method="$1" use_loc="$2"; shift 2
  local jobname="stateest_${MODEL}_${method}"
  local export_str="ALL"
  local kv
  for kv in \
    "NX=${NX}" "NY=${NY}" "NZ=${NZ}" \
    "ENSEMBLE_SIZE=${ENSEMBLE_SIZE}" "NUM_ESMDA_STEPS=${NUM_ESMDA_STEPS}" \
    "INTERVAL_SECONDS=${INTERVAL_SECONDS}" \
    "SIMULATION_TIME=${SIMULATION_TIME}" "NUM_TIME_POINTS=${NUM_TIME_POINTS}" \
    "NUM_ASSIM_WINDOWS=${NUM_ASSIM_WINDOWS}" \
    "ESMDA_SMOOTHER=state_and_dynamic" \
    "USE_LOCALIZATION=${use_loc}" \
    "RUN_SUFFIX=_${method}"
  do export_str="${export_str},${kv}"; done
  echo "  -> ${jobname}  time=${WALLTIME}  partition=${PARTITION} cpus=${CORES_REQ}"
  sbatch \
    --job-name="${jobname}" \
    --partition="${PARTITION}" \
    --time="${WALLTIME}" \
    --cpus-per-task="${CORES_REQ}" \
    --export="${export_str}" \
    "${RUNNER}" "$@" "${FORWARD[@]}"
}

echo "DELFTBLUE [${MODEL}] STATE-ESTIMATION sweep -- methods: ${METHODS}"
echo "Window=${SIMULATION_TIME}s x ${NUM_ASSIM_WINDOWS} windows, ${NUM_TIME_POINTS} knots/window, obs interval=${INTERVAL_SECONDS}s"
echo "Grid=${NX}x${NY}x${NZ} ensemble=${ENSEMBLE_SIZE} steps=${NUM_ESMDA_STEPS}"
[ "${#FORWARD[@]}" -gt 0 ] && echo "Forwarding to every job: ${FORWARD[*]}"

for method in ${METHODS}; do
  case "${method}" in
    corr_ic)
      submit_method corr_ic correlation
      ;;
    dist_ic)
      submit_method dist_ic distance
      ;;
    svd_ic)
      submit_method svd_ic false \
        "esmda/state_reduction=svd" \
        "esmda.state_reduction.basis_source=window_snapshots" \
        "esmda.state_reduction.energy_fraction=${SVD_ENERGY}"
      ;;
    svd_all)
      submit_method svd_all false \
        "esmda/state_reduction=svd" \
        "esmda.state_reduction.basis_source=window_snapshots" \
        "esmda.state_reduction.energy_fraction=${SVD_ENERGY}" \
        "esmda.final_time_smoothing=true"
      ;;
    *)
      echo "error: unknown method '${method}' (expected corr_ic|dist_ic|svd_ic|svd_all)" >&2
      exit 1
      ;;
  esac
done

echo
echo "[${MODEL}] state-estimation sweep submitted -- check with: squeue -u \"\${USER}\""
