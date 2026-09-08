#!/usr/bin/env bash
# Filtering state-reduction benchmark campaign (docs/temp/filtering_state_reduction_benchmark.md).
#
# Runs the seven-run comparison matrix -- full EnKF vs current-cycle SVD at full
# statistical rank, three energy thresholds and two hard rank caps -- through the
# three-stage filtering pipeline, sharing one truth artifact, seed and ensemble.
#
# Concurrency. Runs are pulled from a work queue by NUM_LANES worker lanes
# (default 3), so a slow run never idles the others and the campaign self-
# balances. Each lane gets its OWN `paths.experiment_dir`, which is the only
# shared mutable state a uDALES run has: the truth/template experiment
# (`experiment/999`), the per-member ensemble dirs
# (`ensemble_experiments/NNN`) and the solver output dirs (`outputs/NNN`) all
# nest under it. Two runs sharing one would corrupt each other's namoptions and
# fielddumps. `paths.results_dir` is per-RUN and never shared.
#
#   WALL TIME IS CONTENDED. With NUM_LANES>1 the runs compete for DRAM
#   bandwidth, so end-to-end wall time is NOT comparable across runs and must
#   not be reported as a cost result. Peak RSS (per-run, from /usr/bin/time -v)
#   is unaffected and stays valid. For the campaign's cost question use the
#   per-cycle `analysis_time` / `reduction_basis_time` in cycle_diagnostics.yaml,
#   which are recorded on the reduced AND unreduced paths -- or re-measure the
#   pair of interest with NUM_LANES=1. progress.tsv records each run's start and
#   end timestamps so co-residency can be reconstructed exactly.
#
# Resumability. A finished run writes `<id>.ok`; re-running the driver skips it.
# Each stage also marks itself (`<id>.filter_ok`, `<id>.metrics_ok`), so a
# restart re-runs only what did not finish -- never a completed multi-hour
# filter just because the cheap metrics or figure stage failed.
# A run that fails records its exit code in progress.tsv and the lane moves on --
# losing the whole campaign because one run died is the worst outcome. Stale
# claims from a killed driver are cleared at startup (guarded by a driver-wide
# flock, so a second driver cannot race a live one).
#
# Usage:
#   TRUTH_DIR=<dir with state.nc/params.nc> \
#   BENCH_ROOT=.temp/filtering_state_reduction_benchmark \
#   setsid nohup bash job_scripts/local/filtering_state_reduction_benchmark.sh \
#     > <BENCH_ROOT>/_logs/driver.log 2>&1 < /dev/null &
#
# Env knobs: NUM_LANES (3), WORKERS (8, per run), NUM_CYCLES (60),
# ENSEMBLE_SIZE (50), ONLY (space-separated run ids to restrict the queue).
set -uo pipefail

cd "$(dirname "$0")/../.."
REPO="$(pwd)"

: "${TRUTH_DIR:?set TRUTH_DIR to the directory holding the shared state.nc/params.nc}"
: "${BENCH_ROOT:?set BENCH_ROOT to the campaign output root}"
NUM_LANES="${NUM_LANES:-3}"
WORKERS="${WORKERS:-8}"
NUM_CYCLES="${NUM_CYCLES:-60}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-50}"

LOGS="${REPO}/${BENCH_ROOT}/_logs"
SCRATCH="${REPO}/${BENCH_ROOT}/_scratch"
mkdir -p "${LOGS}" "${SCRATCH}"

if [[ ! -f "${TRUTH_DIR}/state.nc" || ! -f "${TRUTH_DIR}/params.nc" ]]; then
  echo "FATAL: ${TRUTH_DIR} has no state.nc/params.nc" >&2
  exit 1
fi

# Keep JAX off the GPU. run_filtering.py runs under `pixi run -e cuda`, so
# WITHOUT this every filter parent AND every forkserver worker tries to create a
# CUDA context (~264 MB per worker, GBs per parent). On a shared 24 GB card that
# exhausts device memory, a worker is killed mid-forecast, and the filter dies
# with a bare `BrokenProcessPool` hours in -- which is exactly how the first
# attempt at this campaign lost 5 of 7 runs.
#
# Nothing here wants the GPU: uDALES is CPU Fortran, and the analysis is a
# (rows x 50) SVD that CPU JAX handles fine. Pinning the backend also makes the
# runs numerically comparable -- otherwise whichever run happened to win the race
# for device memory would use a different backend from the ones that fell back.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Non-reduction settings, identical for every run. domain.nz and the time knobs
# are pinned explicitly rather than inherited from conf/case/xie_and_castro.yaml,
# so the campaign reproduces from a clean checkout regardless of the working
# tree. time.simulation_time == obs.interval_seconds (60s): one observation
# interval per analysis cycle.
COMMON=(
  case=xie_and_castro
  model@truth_model=pyudales
  model@assim_model=pyudales
  params@truth_params=static_truth
  params@prior_params=static
  "run.truth_dir=${TRUTH_DIR}"
  filtering.mode=joint
  "filtering.num_cycles=${NUM_CYCLES}"
  filtering/analysis=stochastic
  filtering/localization=none
  filtering/inflation=rtps
  filtering/evolution=random_walk
  filtering.seed=42
  "ensemble.ensemble_size=${ENSEMBLE_SIZE}"
  "ensemble.num_parallel_processes=${WORKERS}"
  ensemble.num_cpus_per_process=1
  run.ensemble_save_on_disk=false
  domain.nz=24
  time.simulation_time=60.0
  time.output_frequency=5.0
  # Synthetic turbulent inlet on BOTH mounts: a richer, less smooth flow gives
  # the state's singular spectrum a longer tail, so truncation levels actually
  # separate instead of every setting retaining the same handful of modes. It
  # also switches uDALES from volume nudging to the precursor/driver route
  # (BCxm=3, idriver=2) -- the truth artifact must be generated with the same
  # flag or the observations come from a differently-forced flow.
  #
  # The shipped length scales (50/25/25 m) are too large for this inlet plane:
  # the generator zeroes the plane mean of u', and at L_z=25 m over a 32 m domain
  # (78%) that discards up to ~50% of the fluctuation energy, so the realised rms
  # falls well short of intensity*|U(z)| -- uDALES warns about exactly this.
  # L=(50, 10, 8) keeps the spanwise scale at 12% of the 80 m span and the
  # vertical at 25% of the 32 m height (~half the tallest 18 m roof), which is
  # inside the regime where the mean subtraction is harmless. Pinned here rather
  # than edited into conf/model/pyudales.yaml so the other cases (barcelona,
  # different domains) keep their own calibration.
  truth_model.forward_model.inlet_turbulence.enabled=true
  truth_model.forward_model.inlet_turbulence.length_scale_x=50.0
  truth_model.forward_model.inlet_turbulence.length_scale_y=10.0
  truth_model.forward_model.inlet_turbulence.length_scale_z=8.0
  assim_model.forward_model.inlet_turbulence.enabled=true
  assim_model.forward_model.inlet_turbulence.length_scale_x=50.0
  assim_model.forward_model.inlet_turbulence.length_scale_y=10.0
  assim_model.forward_model.inlet_turbulence.length_scale_z=8.0
)

# id -> reduction overrides. Ordered so the full-rank equivalence CONTROL pair
# (full, svd_full_rank) is claimed first: every truncated run is only
# interpretable once that control passes, so it must survive an early abort.
RUN_IDS=(
  full
  svd_full_rank
  svd_energy_095
  svd_energy_099
  svd_energy_090
  svd_rank_08
  svd_rank_04
)
declare -A RUN_ARGS=(
  [full]="filtering/state_reduction=none"
  [svd_full_rank]="filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=1.0 filtering.state_reduction.max_rank=null"
  [svd_energy_090]="filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=0.90 filtering.state_reduction.max_rank=null"
  [svd_energy_095]="filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=0.95 filtering.state_reduction.max_rank=null"
  [svd_energy_099]="filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=0.99 filtering.state_reduction.max_rank=null"
  [svd_rank_04]="filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=1.0 filtering.state_reduction.max_rank=4"
  [svd_rank_08]="filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=1.0 filtering.state_reduction.max_rank=8"
)

if [[ -n "${ONLY:-}" ]]; then
  read -r -a RUN_IDS <<< "${ONLY}"
fi

# --- Driver-wide lock: refuse to start a second driver over a live one, and
# make the stale-claim sweep below safe (nothing else can be running). ---
exec 9> "${LOGS}/driver.lock"
if ! flock -n 9; then
  echo "FATAL: another driver holds ${LOGS}/driver.lock" >&2
  exit 1
fi

for id in "${RUN_IDS[@]}"; do
  if [[ -d "${LOGS}/${id}.claim" && ! -f "${LOGS}/${id}.ok" ]]; then
    echo "$(date -Is) RECLAIM ${id} (stale claim from a previous driver)"
    rmdir "${LOGS}/${id}.claim"
  fi
done

if [[ ! -f "${LOGS}/progress.tsv" ]]; then
  printf 'id\tlane\trc\tstarted\tended\telapsed_s\n' > "${LOGS}/progress.tsv"
fi

run_one () {
  local id="$1" lane="$2"
  local args="${RUN_ARGS[$id]:-}"
  if [[ -z "${args}" ]]; then
    echo "$(date -Is) lane${lane} UNKNOWN RUN ${id}" >&2
    return 0
  fi

  local t0 t1 rc started ended run_dir
  run_dir="${BENCH_ROOT}/${id}"
  started="$(date -Is)"; t0="$(date +%s)"
  rc=0
  echo "$(date -Is) lane${lane} START ${id}"

  # The three pipeline stages are invoked separately rather than through
  # scripts/run_filtering_pipeline.sh, for two reasons: each gets its own
  # completion marker, so a restart never re-runs a finished multi-hour filter
  # because the cheap metrics or figure stage failed; and /usr/bin/time wraps the
  # FILTER alone, making peak RSS a property of the filter rather than of
  # matplotlib. The run dir is known here (paths.results_dir is set below), so
  # the wrapper's compose-based run-dir resolution is not needed.
  if [[ ! -f "${LOGS}/${id}.filter_ok" ]]; then
    # `args` is word-split on purpose: each element is one Hydra override.
    # shellcheck disable=SC2086
    /usr/bin/time -v -o "${LOGS}/${id}.time" \
      pixi run -e cuda python scripts/filtering/run_filtering.py \
        "${COMMON[@]}" ${args} \
        "paths.results_dir=${run_dir}" \
        "paths.experiment_dir=${SCRATCH}/lane${lane}" \
      > "${LOGS}/${id}.log" 2>&1
    rc=$?
    (( rc == 0 )) && : > "${LOGS}/${id}.filter_ok"
  else
    echo "$(date -Is) lane${lane} SKIP filter ${id} (already done)"
  fi

  if (( rc == 0 )) && [[ ! -f "${LOGS}/${id}.metrics_ok" ]]; then
    pixi run -e dev python scripts/filtering/compute_filtering_metrics.py \
      --run-dir "${run_dir}" >> "${LOGS}/${id}.log" 2>&1
    rc=$?
    (( rc == 0 )) && : > "${LOGS}/${id}.metrics_ok"
  fi

  if (( rc == 0 )); then
    pixi run -e dev python scripts/filtering/make_filtering_figures.py \
      --run-dir "${run_dir}" >> "${LOGS}/${id}.log" 2>&1
    rc=$?
  fi

  t1="$(date +%s)"; ended="$(date -Is)"
  printf '%s\t%d\t%d\t%s\t%s\t%d\n' \
    "${id}" "${lane}" "${rc}" "${started}" "${ended}" "$((t1 - t0))" \
    >> "${LOGS}/progress.tsv"
  if (( rc == 0 )); then
    : > "${LOGS}/${id}.ok"
  else
    echo "$(date -Is) lane${lane} FAILED ${id} rc=${rc} (continuing)" >&2
  fi
  echo "$(date -Is) lane${lane} END ${id} rc=${rc} elapsed=$((t1 - t0))s"
  return 0
}

# One lane: walk the queue, claim what nobody else has taken, run it. `mkdir` is
# the atomic claim -- it succeeds for exactly one lane.
lane_worker () {
  local lane="$1" id
  for id in "${RUN_IDS[@]}"; do
    [[ -f "${LOGS}/${id}.ok" ]] && continue
    mkdir "${LOGS}/${id}.claim" 2>/dev/null || continue
    run_one "${id}" "${lane}"
  done
  echo "$(date -Is) lane${lane} drained"
}

echo "$(date -Is) CAMPAIGN START lanes=${NUM_LANES} workers/run=${WORKERS} cycles=${NUM_CYCLES} ensemble=${ENSEMBLE_SIZE}"
echo "$(date -Is) truth=${TRUTH_DIR}"
echo "$(date -Is) queue=${RUN_IDS[*]}"

pids=()
for (( lane=1; lane<=NUM_LANES; lane++ )); do
  mkdir -p "${SCRATCH}/lane${lane}"
  lane_worker "${lane}" &
  pids+=("$!")
  # Stagger lane starts: the first thing each run does is compile and run the
  # uDALES IBM preprocessing, and three simultaneous gfortran builds writing
  # into freshly created scratch dirs is a needless burst.
  sleep 20
done
for pid in "${pids[@]}"; do wait "${pid}"; done

echo "$(date -Is) CAMPAIGN DONE"
column -t -s $'\t' "${LOGS}/progress.tsv" 2>/dev/null || cat "${LOGS}/progress.tsv"
