#!/bin/bash
# LOCAL (no SLURM) rollout-ESMDA-from-truth runner -- neural-surrogate backend.
#
# Runs scripts/run_esmda.py DIRECTLY (no sbatch / module / SLURM env vars),
# keeping all heavy I/O and outputs under the repo (pyurbanair). Run config that
# is shared with the pyudales/pylbm/pypalm runners lives in ../common.sh (sourced
# below); only the neural-surrogate specifics are set here. The point of this
# runner is to assimilate the EXACT same experiment as the other backends -- same
# ground truth, domain bounds, windows, time horizon, sensors and dynamic
# parameters (all from COMMON_RUN_FLAGS) -- with the trained surrogate as the
# assimilation model.
#
# Hybrid execution: the surrogate's per-member cold-start SPIN-UP is the pyudales
# CPU ensemble, fanned out across parallel processes (ensemble.num_parallel_processes
# = min(ENSEMBLE_SIZE, LOCAL_MAX_PARALLEL), the max you choose -- default 16, see
# common.sh). The network ROLLOUT is then a single batched pass on the GPU, so a
# visible NVIDIA GPU and the `cuda` pixi env are required.
#
# FIXED GRID: the surrogate only runs at the resolution it was trained on
# (60x80x16, which is also the ground-truth grid). NX/NY/NZ are therefore PINNED
# here -- a value injected from the environment that disagrees is overridden back
# to the trained grid with a warning. This means a *domain* (resolution) sweep is
# meaningless for the surrogate; the ensemble / esmda-steps / interval sweeps are
# the ones that apply (they already hold the grid fixed at 60x80x16).
#
# Run it from anywhere (it cd's to the repo root itself):
#
#     bash job_scripts/local/neural_surrogate/rollout_esmda_from_truth.sh
#
# Extra Hydra overrides may be appended and take precedence, e.g.:
#
#     bash job_scripts/local/neural_surrogate/rollout_esmda_from_truth.sh esmda.num_steps=4
#
# ENSEMBLE_SIZE, NUM_ESMDA_STEPS and INTERVAL_SECONDS are read from the
# environment (the sweep launchers in this folder set them), so each
# configuration lands in its own RESULTS_DIR. MODEL_DIR selects the trained
# weights folder (default model_weights/unet_convnext_medium).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Backend-specific knobs ------------------------------------------------------
ENV="${ENV:-cuda}"             # GPU pixi env (the surrogate rollout runs on CUDA)
ASSIM_MODEL="neural_surrogate"

# Split the devices: torch runs the surrogate network on the GPU (device=cuda),
# but JAX (the ESMDA linear algebra) must stay on the CPU. Otherwise every one of
# the parallel spin-up worker processes re-imports the ESMDA module, whose
# module-level jax.random.PRNGKey() default forces JAX to grab a CUDA backend in
# each worker -- the workers then preallocate GPU memory on top of torch and the
# run dies with CUDA_ERROR_OUT_OF_MEMORY / BrokenProcessPool. The `cuda` pixi env
# does not set this (only the `cpu` feature does), so pin it here. Exported so it
# reaches the forkserver/spawn workers.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Trained-weights folder (written by scripts/neural_surrogate/train_neural_surrogate.py). Its
# config.yaml supplies the architecture/state_vars; weights.pt supplies the
# parameters. Override with MODEL_DIR=… . NB: the conf default points at
# unet_convnext_small, which is not present locally -- this runner defaults to the
# medium model that is.
MODEL_DIR="${MODEL_DIR:-model_weights/unet_convnext_medium}"

# Fixed trained grid -- the surrogate ONLY runs at this resolution. Anything the
# environment injects (e.g. a domain sweep) is coerced back to it.
TRAINED_NX=60
TRAINED_NY=80
TRAINED_NZ=16
NX="${NX:-${TRAINED_NX}}"
NY="${NY:-${TRAINED_NY}}"
NZ="${NZ:-${TRAINED_NZ}}"
if [ "${NX}" != "${TRAINED_NX}" ] || [ "${NY}" != "${TRAINED_NY}" ] || [ "${NZ}" != "${TRAINED_NZ}" ]; then
  echo "note: neural surrogate only runs at its trained grid ${TRAINED_NX}x${TRAINED_NY}x${TRAINED_NZ};" \
       "overriding requested ${NX}x${NY}x${NZ}." >&2
  NX="${TRAINED_NX}"; NY="${TRAINED_NY}"; NZ="${TRAINED_NZ}"
fi

# Sweep parameters. Env-overridable so the sweep launchers can inject one value
# per run; each lands in its own RESULTS_DIR.
NUM_ESMDA_STEPS="${NUM_ESMDA_STEPS:-3}"
NUM_PARALLEL="${NUM_PARALLEL:-}"    # empty -> min(ENSEMBLE_SIZE, LOCAL_MAX_PARALLEL)
# ENSEMBLE_SIZE and INTERVAL_SECONDS default in common.sh (still env-overridable).

# Shared defaults: paths, domain bounds + sensors, windows, time horizon, dynamic
# parameter settings, localization, ground-truth resolution/validation, and the
# COMMON_RUN_FLAGS array of every shared Hydra override.
source "${REPO_ROOT}/job_scripts/local/common.sh"

# Worker count for the SPIN-UP ensemble: fan the per-member cold starts out across
# up to LOCAL_MAX_PARALLEL parallel pyudales processes -- the maximum YOU choose
# (default 16; set it in common.sh or per run via LOCAL_MAX_PARALLEL=…). Capped at
# the ensemble size so no worker sits idle; pin an exact count with NUM_PARALLEL=…
# The network rollout itself is a single batched GPU pass regardless.
if [ -z "${NUM_PARALLEL}" ]; then
  NUM_PARALLEL=$(( ENSEMBLE_SIZE < LOCAL_MAX_PARALLEL ? ENSEMBLE_SIZE : LOCAL_MAX_PARALLEL ))
fi

RUN_TAG="${ASSIM_MODEL}_nx${NX}_ny${NY}_nz${NZ}_ens${ENSEMBLE_SIZE}_steps${NUM_ESMDA_STEPS}_int${INTERVAL_SECONDS}${LOCALIZATION_TAG}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_TAG}"
RUN_TEMP_DIR="${TEMP_ROOT}/${RUN_TAG}_$$"

mkdir -p "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "LOCAL neural-surrogate rollout-ESMDA on $(hostname) -- $(date)"
echo "Truth=${GROUND_TRUTH_MODEL} (loaded) assim=${ASSIM_MODEL} case=${CASE} domain=${NX}x${NY}x${NZ}"
echo "Model weights: ${MODEL_DIR}"
echo "Ground truth: ${GROUND_TRUTH_PATH}"
echo "Output: ${RESULTS_DIR}  (temp: ${RUN_TEMP_DIR})"
echo "Ensemble=${ENSEMBLE_SIZE} spinup-parallel=${NUM_PARALLEL} (GPU rollout) windows=${NUM_ASSIM_WINDOWS}"
echo "ESMDA steps=${NUM_ESMDA_STEPS} obs_interval=${INTERVAL_SECONDS}s localization=${USE_LOCALIZATION}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

# Surrogate-specific flags:
#  * model_dir / device   -- which trained net, and run its rollout on the GPU.
#  * spinup_forward_model.temp_dir / output_dir -- keep the per-member pyudales
#    cold-start scratch inside this run's private temp dir (auto-purged on success).
#  * run.ensemble_save_on_disk=true -- stream the ensemble forecasts one NetCDF per
#    member instead of concatenating the full field in memory, matching the other
#    backends (run_esmda.py reassembles per-window states from the per-member files).
EXTRA_FLAGS=(
  "assim_model.forward_model.model_dir=${MODEL_DIR}"
  "assim_model.forward_model.device=cuda"
  "+assim_model.forward_model.spinup_forward_model.temp_dir=${RUN_TEMP_DIR}"
  "+assim_model.forward_model.spinup_forward_model.output_dir=${RUN_TEMP_DIR}/outputs"
  "run.ensemble_save_on_disk=true"
)

# COMMON_RUN_FLAGS (from common.sh) carries every shared Hydra override; only the
# assim model, the per-run sweep values, hydra.run.dir and the surrogate flags are
# added here -- so this run is the EXACT same experiment as the other backends.
pixi run -e "${ENV}" -- python -u \
    scripts/run_esmda.py \
    "${COMMON_RUN_FLAGS[@]}" \
    model@assim_model="${ASSIM_MODEL}" \
    domain.nx="${NX}" \
    domain.ny="${NY}" \
    domain.nz="${NZ}" \
    ensemble.ensemble_size="${ENSEMBLE_SIZE}" \
    ensemble.num_parallel_processes="${NUM_PARALLEL}" \
    esmda.num_steps="${NUM_ESMDA_STEPS}" \
    obs.interval_seconds="${INTERVAL_SECONDS}" \
    "hydra.run.dir=${RESULTS_DIR}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "Done -- rollout ESMDA outputs under ${RESULTS_DIR}"
