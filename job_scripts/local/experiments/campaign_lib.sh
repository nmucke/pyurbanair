#!/usr/bin/env bash
# Campaign machinery shared by the three per-method runners. Knobs live in
# settings.sh; this file only turns them into runs.
#
# Contract for a runner:
#
#   source settings.sh; source campaign_lib.sh
#   campaign_init <method>                  # dirs, lock, common args, guards
#   enqueue <run-id> <hydra override>...    # once per axis combination
#   drain <pipeline script>                 # run the queue
#
# Resumability. A finished run writes `_logs/<id>.ok` and is skipped on a
# re-run; a failed one records its exit code in `_logs/progress.tsv` and the
# lane moves on. Unlike the per-stage benchmark drivers, a run is one call to
# the three-stage pipeline script, so an interrupted run restarts from its
# assimilation stage — the markers are per run, not per stage.

# --- Small helpers ----------------------------------------------------------

fatal () {
  echo "FATAL: $*" >&2
  exit 1
}

# Filename-safe tag for a numeric axis value: 30.0 -> 30, 12.5 -> 12p5.
tag () {
  local v="$1"
  v="${v%.0}"
  printf '%s' "${v//./p}"
}

# Join the axis tags into a run id.
run_id () {
  local IFS=_
  printf '%s' "$*"
}

# Resolve PALM's absolute disturbance amplitude from the shared intensity, so
# `inflow_turb` targets the same nominal rms on both backends. An explicit
# PALM_DISTURBANCE_AMPLITUDE wins. Sets the global the pypalm branch of
# inflow_overrides reads; call before it.
#
# PALM draws uniform noise on [-1.5A, 1.5A] (disturb_field.f90:119), rms =
# A*sqrt(3)/2, so A = intensity*U_ref / (sqrt(3)/2) targets the same rms uDALES
# aims at with intensity*|U(z)|. BOTH realise something other than nominal —
# PALM smooths the field twice before ADDING it (so repeated kicks accumulate),
# uDALES subtracts the plane mean — so treat this as equal nominal forcing, not
# equal measured rms. The kick INTERVAL is deliberately not derived; see the
# PALM_DT_DISTURB comment in settings.sh.
resolve_inlet_turbulence () {
  PALM_AMPLITUDE_RESOLVED="${PALM_DISTURBANCE_AMPLITUDE}"
  if [[ -z "${PALM_AMPLITUDE_RESOLVED}" ]]; then
    PALM_AMPLITUDE_RESOLVED="$(awk -v i="${INLET_INTENSITY}" -v u="${INLET_REFERENCE_SPEED}" \
      'BEGIN { printf "%.4f", i * u / (sqrt(3) / 2) }')"
  fi
  PALM_DT_DISTURB_RESOLVED="${PALM_DT_DISTURB}"
}

# Hydra overrides putting ONE model mount into one INFLOW axis setting, one per
# line. The three settings mean the same thing physically in both backends, but
# the knobs behind them do not — uDALES' digital-filter driver planes and PALM's
# random inflow disturbances are different mechanisms — so the mapping is per
# backend and the mount is explicit (the truth and assimilation mounts can now
# hold different backends).
inflow_overrides () {
  local backend="$1" mount="$2" setting="$3"
  local -a out=()

  case "${backend}" in
  pyudales)
    case "${setting}" in
    inflow)
      out+=(
        "${mount}.forward_model.boundary_condition=inflow_outflow"
        "${mount}.forward_model.inlet_turbulence.enabled=false"
      )
      ;;
    inflow_turb)
      # The backend turns volume nudging off under the driver route (BCxm=3):
      # relaxing the interior toward a smooth mean would damp the injected
      # fluctuations. That is the intended behaviour, not a missing override.
      out+=(
        "${mount}.forward_model.boundary_condition=inflow_outflow"
        "${mount}.forward_model.inlet_turbulence.enabled=true"
        "${mount}.forward_model.inlet_turbulence.intensity=${INLET_INTENSITY}"
        "${mount}.forward_model.inlet_turbulence.length_scale_x=${UDALES_INLET_LENGTH_SCALE_X}"
        "${mount}.forward_model.inlet_turbulence.length_scale_y=${UDALES_INLET_LENGTH_SCALE_Y}"
        "${mount}.forward_model.inlet_turbulence.length_scale_z=${UDALES_INLET_LENGTH_SCALE_Z}"
        "${mount}.forward_model.inlet_turbulence.time_step=${UDALES_INLET_TIME_STEP}"
      )
      ;;
    periodic)
      # Under periodic BCs the interior nudging is the ONLY momentum source
      # (the dpdx/dpdy body force is written as zero), so it stays on; the
      # synthetic inlet requires inflow_outflow and is off.
      out+=(
        "${mount}.forward_model.boundary_condition=periodic"
        "${mount}.forward_model.inlet_turbulence.enabled=false"
        "${mount}.forward_model.nudging_config.interior_nudging=true"
      )
      ;;
    *) fatal "unknown INFLOW setting '${setting}' (expected: inflow inflow_turb periodic)" ;;
    esac
    ;;

  pypalm)
    # PALM_NCPU applies to every setting (it is a decomposition knob, not a
    # forcing one) and must divide domain.nx — see settings.sh.
    out+=("${mount}.forward_model.ncpu=${PALM_NCPU}")
    case "${setting}" in
    inflow)
      # `initial_seed` is PALM's cold-start symmetry-breaking kick, a DIFFERENT
      # mechanism from inlet turbulence, so it follows its own knob here rather
      # than being forced off with the inlet.
      out+=(
        "${mount}.forward_model.boundary_condition=inflow_outflow"
        "${mount}.forward_model.inlet_turbulence.enabled=false"
        "${mount}.forward_model.inlet_turbulence.initial_seed=${PALM_INITIAL_SEED}"
      )
      ;;
    inflow_turb)
      # Random u/v perturbations re-injected in the inflow strip for the whole
      # run, paced by dt_disturb (PALM's only toggleable inlet-turbulence
      # mechanism). The initial kick is forced ON: injecting fluctuations into a
      # perfectly symmetric analytic profile otherwise wastes the spin-up.
      # Amplitude and cadence are derived from the SHARED intensity / length
      # scale (see resolve_inlet_turbulence and settings.sh), and the strip is
      # placed to match uDALES' inlet-plane injection as closely as PALM's
      # mechanism allows.
      out+=(
        "${mount}.forward_model.boundary_condition=inflow_outflow"
        "${mount}.forward_model.inlet_turbulence.enabled=true"
        "${mount}.forward_model.inlet_turbulence.initial_seed=true"
        "${mount}.forward_model.inlet_turbulence.amplitude=${PALM_AMPLITUDE_RESOLVED}"
        "${mount}.forward_model.inlet_turbulence.dt_disturb=${PALM_DT_DISTURB_RESOLVED}"
      )
      [[ -n "${PALM_DISTURBANCE_BEGIN}" ]] &&
        out+=("${mount}.forward_model.inlet_turbulence.begin=${PALM_DISTURBANCE_BEGIN}")
      [[ -n "${PALM_DISTURBANCE_END}" ]] &&
        out+=("${mount}.forward_model.inlet_turbulence.end=${PALM_DISTURBANCE_END}")
      [[ -n "${PALM_DISTURBANCE_LEVEL_B}" ]] &&
        out+=("${mount}.forward_model.inlet_turbulence.level_b=${PALM_DISTURBANCE_LEVEL_B}")
      [[ -n "${PALM_DISTURBANCE_LEVEL_T}" ]] &&
        out+=("${mount}.forward_model.inlet_turbulence.level_t=${PALM_DISTURBANCE_LEVEL_T}")
      ;;
    periodic)
      # PALM REJECTS inlet_turbulence.enabled=true under cyclic BCs at
      # construction (it derives no inflow strip), so the inlet is off and the
      # nudging driver — the counterpart of uDALES' interior nudging, same
      # relaxation physics and the same parameter meaning — is what drives the
      # domain. Without it a cyclic PALM run is un-driven and decays.
      out+=(
        "${mount}.forward_model.boundary_condition=periodic"
        "${mount}.forward_model.inlet_turbulence.enabled=false"
        "${mount}.forward_model.inlet_turbulence.initial_seed=${PALM_INITIAL_SEED}"
        "${mount}.forward_model.nudging_config.enabled=true"
      )
      ;;
    *) fatal "unknown INFLOW setting '${setting}' (expected: inflow inflow_turb periodic)" ;;
    esac
    ;;

  *)
    fatal "no INFLOW mapping for backend '${backend}' (expected: pyudales pypalm)"
    ;;
  esac

  printf '%s\n' "${out[@]}"
}

# Spin-up plateau for one INFLOW setting. Periodic runs build their momentum
# field from rest through the nudging relaxation; the inflow_outflow ones are
# driven by the inlet from the first step. See settings.sh.
spinup_for () {
  case "$1" in
  inflow | inflow_turb) printf '%s' "${SPINUP_TIME_INFLOW}" ;;
  periodic) printf '%s' "${SPINUP_TIME_PERIODIC}" ;;
  *) fatal "no spin-up mapping for INFLOW setting '$1'" ;;
  esac
}

# The model half of one run: which backend is mounted as the truth, both mounts'
# forcing, and the spin-up that forcing needs. The assimilation mount is
# ASSIM_MODEL (a constant) and is already in COMMON_ARGS, so only its forcing is
# emitted here.
model_overrides () {
  local truth="$1" setting="$2" spinup
  spinup="$(spinup_for "${setting}")" || exit 1
  printf 'model@truth_model=%s\n' "${truth}"
  printf 'time.spinup_time=%s\n' "${spinup}"
  inflow_overrides "${truth}" truth_model "${setting}"
  inflow_overrides "${ASSIM_MODEL}" assim_model "${setting}"
}

# Cycles must tile a window: SIMULATION_TIME / OUTPUT_FREQUENCY must be a whole
# number of observation frames, and `stride` must divide it. Both filter-bearing
# entry points refuse to start otherwise — catch it here instead of after the
# first ensemble has been built.
check_cycle_tiling () {
  local stride="$1"
  awk -v s="${SIMULATION_TIME}" -v o="${OUTPUT_FREQUENCY}" -v n="${stride}" 'BEGIN {
    frames = s / o
    if (frames != int(frames + 0.5) || (frames - int(frames + 0.5)) > 1e-9) exit 1
    frames = int(frames + 0.5)
    if (frames % n != 0) exit 2
  }' || fatal "SIMULATION_TIME=${SIMULATION_TIME} / OUTPUT_FREQUENCY=${OUTPUT_FREQUENCY}" \
    "must be a whole number of frames, divisible by the analysis stride ${stride}"
}

# --- Campaign setup ---------------------------------------------------------

# Overrides identical across every run of every campaign. Built by
# campaign_init so the runners can splice it in verbatim.
declare -a COMMON_ARGS=()
declare -a RUN_IDS=()
declare -A RUN_ARGS=()

campaign_init () {
  METHOD="$1"
  REPO="$(pwd)"

  [[ "${RESULTS_ROOT}" = /* ]] || RESULTS_ROOT="${REPO}/${RESULTS_ROOT}"
  CAMPAIGN_ROOT="${RESULTS_ROOT}/${METHOD}"
  LOGS="${CAMPAIGN_ROOT}/_logs"
  SCRATCH="${CAMPAIGN_ROOT}/_scratch"
  mkdir -p "${LOGS}" "${SCRATCH}"

  # A shared truth artifact is BOTH forcing- and model-specific: it was
  # simulated with one boundary condition / inlet setting, by one backend. It
  # cannot serve a sweep over either.
  if [[ -n "${TRUTH_DIR}" ]]; then
    [[ -f "${TRUTH_DIR}/state.nc" && -f "${TRUTH_DIR}/params.nc" ]] ||
      fatal "TRUTH_DIR=${TRUTH_DIR} has no state.nc/params.nc"
    local n_inflow n_truth
    n_inflow="$(wc -w <<< "${INFLOW_LIST}")"
    n_truth="$(wc -w <<< "${TRUTH_MODEL_LIST}")"
    ((n_inflow == 1)) || fatal "TRUTH_DIR is set but INFLOW_LIST has ${n_inflow} values;" \
      "a truth artifact carries the forcing it was simulated with. Run one" \
      "inflow setting per TRUTH_DIR, or unset TRUTH_DIR to simulate inline."
    ((n_truth == 1)) || fatal "TRUTH_DIR is set but TRUTH_MODEL_LIST has ${n_truth} values;" \
      "a truth artifact is the output of ONE backend. Run one truth model per" \
      "TRUTH_DIR, or unset TRUTH_DIR to simulate inline."
  fi

  resolve_inlet_turbulence

  # Validate the model x inflow axes HERE. inflow_overrides() is consumed
  # through a process substitution, whose exit is the subshell's — an unknown
  # value would otherwise slip through as an empty override list (a run with the
  # config's default forcing/backend, silently off-axis). A command substitution
  # is a subshell too, but its status does reach here.
  local truth setting
  for truth in ${TRUTH_MODEL_LIST}; do
    for setting in ${INFLOW_LIST}; do
      model_overrides "${truth}" "${setting}" > /dev/null || exit 1
    done
  done

  COMMON_ARGS=(
    "case=${CASE}"
    "model@assim_model=${ASSIM_MODEL}"
    "params@truth_params=${TRUTH_PARAMS}"
    "time.simulation_time=${SIMULATION_TIME}"
    "time.output_frequency=${OUTPUT_FREQUENCY}"
    "time.seconds_per_knot=${SECONDS_PER_KNOT}"
    "ensemble.ensemble_size=${ENSEMBLE_SIZE}"
    "ensemble.num_parallel_processes=${WORKERS}"
    ensemble.num_cpus_per_process=1
    "run.skip_viz=${SKIP_VIZ}"
  )
  [[ -n "${TRUTH_DIR}" ]] && COMMON_ARGS+=("run.truth_dir=${TRUTH_DIR}")
  [[ -n "${TRUTH_START_TIME}" ]] && COMMON_ARGS+=("run.truth_start_time=${TRUTH_START_TIME}")
  [[ -n "${PARAMS_TO_ESTIMATE}" ]] && COMMON_ARGS+=("${PARAMS_TO_ESTIMATE}")

  if [[ ! -f "${LOGS}/progress.tsv" ]]; then
    printf 'id\tlane\trc\tstarted\tended\telapsed_s\n' > "${LOGS}/progress.tsv"
  fi

  # Refuse to start a second driver over a live one (and make the stale-claim
  # sweep in drain() safe: nothing else can be running).
  exec 9> "${LOGS}/driver.lock"
  flock -n 9 || fatal "another ${METHOD} campaign holds ${LOGS}/driver.lock"
}

# enqueue <id> <override>...  — every override must be a single shell word.
enqueue () {
  local id="$1" tok
  shift
  for tok in "$@"; do
    [[ "${tok}" == *[[:space:]]* ]] && fatal "override '${tok}' for ${id} contains whitespace"
  done
  [[ -n "${RUN_ARGS[${id}]:-}" ]] && fatal "duplicate run id '${id}'"
  RUN_IDS+=("${id}")
  RUN_ARGS["${id}"]="$*"
}

# --- Execution --------------------------------------------------------------

# Optional function name called with the run dir after a run's pipeline has
# finished SUCCESSFULLY — i.e. after assimilation, metrics, figures and the
# ESMDA-schema view have all read whatever they need. Empty = nothing runs.
: "${POST_RUN_HOOK:=}"

# A POST_RUN_HOOK that deletes state_history.nc once every stage is done with
# it. The filter writes the same object twice: `_save_window_state` puts each
# window's analyzed frames in windows/window_{w}_posterior_state.nc (the
# ESMDA-schema artifact, and the ONLY source for the ESMDA-schema stages) and
# the same pieces are concatenated into state_history.nc (cycle-indexed, for
# the filtering-native per-cycle state RMSE). At one analysis per output frame
# that is ~7 GB of pure duplication per run.
#
# The window files are kept because they have no fallback and are what makes a
# filtering run comparable to an ESMDA one; state_history.nc is rebuildable
# from them:
#   xarray.concat([...window_{w}_posterior_state.nc...], "time").rename(time="cycle")
#
# Refuses to delete unless those window files exist AND their bytes account for
# the history's — never turn the only copy of the analyzed states into none.
prune_state_history () {
  local run_dir="$1"
  local history="${run_dir}/state_history.nc"
  [[ -f "${history}" ]] || return 0

  local -a window_states=()
  shopt -s nullglob
  window_states=("${run_dir}"/windows/window_*_posterior_state.nc)
  shopt -u nullglob
  if ((${#window_states[@]} == 0)); then
    echo "$(date -Is) KEEP state_history.nc: no windows/window_*_posterior_state.nc" \
      "to fall back on in ${run_dir}" >&2
    return 0
  fi

  local history_bytes window_bytes
  history_bytes="$(stat -c %s "${history}")"
  window_bytes="$(stat -c %s "${window_states[@]}" | awk '{s += $1} END {print s}')"
  if ! awk -v w="${window_bytes}" -v h="${history_bytes}" \
    'BEGIN { exit !(w >= 0.95 * h) }'; then
    echo "$(date -Is) KEEP state_history.nc: the ${#window_states[@]} window state" \
      "file(s) total ${window_bytes} B against its ${history_bytes} B, so they do" \
      "not hold the same frames (${run_dir})" >&2
    return 0
  fi

  rm -f "${history}"
  echo "$(date -Is) pruned state_history.nc ($((history_bytes / 1000000)) MB);" \
    "the same analyzed frames remain in ${#window_states[@]} window state file(s)"
}

run_one () {
  local id="$1" lane="$2"
  local args="${RUN_ARGS[${id}]}"
  local run_dir="${CAMPAIGN_ROOT}/${id}"
  local t0 t1 rc started ended

  started="$(date -Is)"
  t0="$(date +%s)"
  echo "$(date -Is) lane${lane} START ${id}"

  # Recorded next to the log so a run dir can always be traced back to the
  # exact override set that produced it.
  printf '%s\n' ${args} "paths.results_dir=${run_dir}" \
    "paths.experiment_dir=${SCRATCH}/lane${lane}" > "${LOGS}/${id}.args"

  # `args` is word-split on purpose: each element is one Hydra override.
  # shellcheck disable=SC2086
  bash "${PIPELINE}" ${args} \
    "paths.results_dir=${run_dir}" \
    "paths.experiment_dir=${SCRATCH}/lane${lane}" \
    > "${LOGS}/${id}.log" 2>&1
  rc=$?

  # Only on a clean run: a failed pipeline keeps every artifact for the
  # post-mortem, and a hook must never be what turns a partial run into an
  # unreadable one.
  if ((rc == 0)) && [[ -n "${POST_RUN_HOOK}" ]]; then
    "${POST_RUN_HOOK}" "${run_dir}" >> "${LOGS}/${id}.log" 2>&1 ||
      echo "$(date -Is) lane${lane} ${POST_RUN_HOOK} failed for ${id} (continuing)" >&2
  fi

  t1="$(date +%s)"
  ended="$(date -Is)"
  printf '%s\t%d\t%d\t%s\t%s\t%d\n' \
    "${id}" "${lane}" "${rc}" "${started}" "${ended}" "$((t1 - t0))" \
    >> "${LOGS}/progress.tsv"
  if ((rc == 0)); then
    : > "${LOGS}/${id}.ok"
  else
    echo "$(date -Is) lane${lane} FAILED ${id} rc=${rc} (see ${LOGS}/${id}.log; continuing)" >&2
  fi
  echo "$(date -Is) lane${lane} END ${id} rc=${rc} elapsed=$((t1 - t0))s"
  return 0
}

# One lane: walk the queue and claim what nobody else has taken. `mkdir` is the
# atomic claim — it succeeds for exactly one lane.
lane_worker () {
  local lane="$1" id
  for id in "${RUN_IDS[@]}"; do
    [[ -f "${LOGS}/${id}.ok" ]] && continue
    mkdir "${LOGS}/${id}.claim" 2>/dev/null || continue
    run_one "${id}" "${lane}"
  done
  echo "$(date -Is) lane${lane} drained"
}

drain () {
  PIPELINE="$1"
  local id lane
  local -a pids=()

  if [[ -n "${ONLY}" ]]; then
    local -a kept=() want
    local sel
    read -r -a want <<< "${ONLY}"
    for id in "${RUN_IDS[@]}"; do
      for sel in "${want[@]}"; do
        [[ "${id}" == "${sel}" ]] && kept+=("${id}") && break
      done
    done
    ((${#kept[@]} > 0)) || fatal "ONLY='${ONLY}' matched none of: ${RUN_IDS[*]}"
    RUN_IDS=("${kept[@]}")
  fi

  echo "$(date -Is) CAMPAIGN ${METHOD}: ${#RUN_IDS[@]} runs, lanes=${NUM_LANES}," \
    "workers/run=${WORKERS}, ensemble=${ENSEMBLE_SIZE}, sim_time=${SIMULATION_TIME}s"
  echo "$(date -Is) output root: ${CAMPAIGN_ROOT}"

  if ((DRY_RUN != 0)); then
    for id in "${RUN_IDS[@]}"; do
      echo "--- ${id}"
      echo "    bash ${PIPELINE} ${RUN_ARGS[${id}]} paths.results_dir=${CAMPAIGN_ROOT}/${id} paths.experiment_dir=${SCRATCH}/lane1"
    done
    echo "$(date -Is) DRY_RUN=1: nothing executed"
    return 0
  fi

  # Clear claims left behind by a killed driver (safe: the flock in
  # campaign_init guarantees no other driver is live).
  for id in "${RUN_IDS[@]}"; do
    if [[ -d "${LOGS}/${id}.claim" && ! -f "${LOGS}/${id}.ok" ]]; then
      echo "$(date -Is) RECLAIM ${id} (stale claim from a previous driver)"
      rmdir "${LOGS}/${id}.claim"
    fi
  done

  for ((lane = 1; lane <= NUM_LANES; lane++)); do
    mkdir -p "${SCRATCH}/lane${lane}"
    lane_worker "${lane}" &
    pids+=("$!")
    # Stagger: the first thing each run does is build and run the uDALES IBM
    # preprocessing, and simultaneous gfortran builds into fresh scratch dirs
    # are a needless burst.
    ((NUM_LANES > 1)) && sleep 20
  done
  for id in "${pids[@]}"; do wait "${id}"; done

  echo "$(date -Is) CAMPAIGN ${METHOD} DONE"
  column -t -s $'\t' "${LOGS}/progress.tsv" 2>/dev/null || cat "${LOGS}/progress.tsv"
}
