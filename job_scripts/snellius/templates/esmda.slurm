#!/bin/bash
# Generic time-varying-parameter rollout ESMDA template for Snellius. NOT meant
# to be `sbatch`ed directly — submit via the wrapper, which fills in
# --job-name, --partition, --cpus-per-task, --time, --output/--error and the
# PUA_* environment:
#
#     job_scripts/snellius/submit.sh <assim_model> <size> [extra hydra overrides...]
#     TRUTH_MODEL=<m> job_scripts/snellius/submit.sh <assim_model> <size> ...
#
# Truth and assimilation forward models are independent (PUA_TRUTH_MODEL /
# PUA_ASSIM_MODEL), so the same template covers single-model runs and twin
# experiments (e.g. truth=pyudales, assimilate with pylbm). Per-model solver
# flags are composed per role below.
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1750M
#SBATCH --account=tdsr72361

set -euo pipefail

# sbatch's --chdir (set by submit.sh) has already put us in the per-job
# workdir on /scratch-shared, populated with symlinks back to the repo. We
# deliberately do NOT `cd "${SLURM_SUBMIT_DIR}"` here: on Snellius that env
# var points at the directory sbatch was invoked from (the shared repo root),
# which would undo the per-job isolation and let concurrent jobs clobber each
# other's `.temp`. If you ever invoke this template outside the wrapper, set
# cwd manually before sbatch.

: "${PUA_SIZE:?PUA_SIZE not set — submit via job_scripts/snellius/submit.sh}"
: "${PUA_NUM_PARALLEL:?PUA_NUM_PARALLEL not set — submit via job_scripts/snellius/submit.sh}"
: "${PUA_TRUTH_MODEL:?PUA_TRUTH_MODEL not set — submit via job_scripts/snellius/submit.sh}"
: "${PUA_ASSIM_MODEL:?PUA_ASSIM_MODEL not set — submit via job_scripts/snellius/submit.sh}"

# Logs -> job_scripts/snellius/out_files; temp/intermediate I/O ->
# /scratch-shared (shared GPFS); final results -> ${PROJECT_STORAGE} on the
# project space.
#
# NB: do NOT point RUN_TEMP_DIR at $TMPDIR (/scratch-local/<jobid>) — pyudales
# same-model runs (truth_model == assim_model == pyudales) fail there: the
# second of the two prepare() calls exits the (silent-stderr) write_inputs.sh
# preprocessor with status 1, consistently in <40s. The exact same workflow
# succeeds when temp_dir lives on /scratch-shared (verified on debug job
# 23170164). Cause not yet pinpointed (pyudales swallows the bash error via
# subprocess.DEVNULL by default); /scratch-shared is the working location and
# the cost is GPFS I/O latency vs node-local. We clean up RUN_TEMP_DIR at
# successful job exit so /scratch-shared doesn't accumulate per-run trees;
# failures leave the dir behind for post-mortem inspection.
PROJECT_STORAGE="/projects/prjs2075/urbanair"
RUN_TEMP_DIR="/scratch-shared/${USER}/urbanair_temp/${SLURM_JOB_ID}"
mkdir -p job_scripts/snellius/out_files "${PROJECT_STORAGE}" "${RUN_TEMP_DIR}"
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Truth=${PUA_TRUTH_MODEL} assim=${PUA_ASSIM_MODEL} size=${PUA_SIZE} cores=${SLURM_CPUS_PER_TASK} num_parallel=${PUA_NUM_PARALLEL}"
echo "Results: ${PROJECT_STORAGE}  (temp: ${RUN_TEMP_DIR})"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# Each ensemble worker pins to one core; keep in-rank BLAS single-threaded so
# workers x BLAS threads don't fight for the allocated cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Stderr is fully buffered when redirected to a file — without this a crash
# mid-run loses the traceback.
export PYTHONUNBUFFERED=1

# Per-role solver flags. `role` is the Hydra group (truth_model | assim_model);
# `model` is the forward model assigned to that role.
role_flags() {
  local role="$1" model="$2"
  case "${model}" in
    pylbm)
      # CPU build: keep the LBM solver off CUDA on the compute partition.
      printf '%s\n' "${role}.forward_model.cuda=false"
      ;;
    pyudales)
      printf '%s\n' "+${role}.forward_model.temp_dir=${RUN_TEMP_DIR}"
      printf '%s\n' "+${role}.forward_model.output_dir=${RUN_TEMP_DIR}/outputs"
      ;;
    pypalm)
      printf '%s\n' "+${role}.forward_model.temp_dir=${RUN_TEMP_DIR}"
      ;;
    *)
      echo "unsupported model '${model}' for role '${role}'" >&2
      exit 1
      ;;
  esac
}

EXTRA_FLAGS=()

# Global: if pypalm is involved in either role, pin the shared vertical
# resolution PALM expects (matches the DelftBlue pypalm scripts).
case " ${PUA_TRUTH_MODEL} ${PUA_ASSIM_MODEL} " in
  *" pypalm "*) EXTRA_FLAGS+=( "domain.nz=16" ) ;;
esac

# Global: pylbm's forward model hardcodes ./.temp relative to cwd. If either
# role is pylbm, point it at node-local scratch via a symlink so intermediate
# I/O stays off the home quota.
case " ${PUA_TRUTH_MODEL} ${PUA_ASSIM_MODEL} " in
  *" pylbm "*)
    rm -rf .temp
    ln -sfn "${RUN_TEMP_DIR}" .temp
    ;;
esac

# Per-role flags for each model.
while IFS= read -r flag; do
  [ -n "${flag}" ] && EXTRA_FLAGS+=( "${flag}" )
done < <(role_flags truth_model "${PUA_TRUTH_MODEL}"; role_flags assim_model "${PUA_ASSIM_MODEL}")

pixi run -e snellius -- python -u scripts/run_time_varying_parameters_rollout_esmda.py \
    size="${PUA_SIZE}" \
    model@truth_model="${PUA_TRUTH_MODEL}" \
    model@assim_model="${PUA_ASSIM_MODEL}" \
    ensemble.num_parallel_processes="${PUA_NUM_PARALLEL}" \
    localization=correlation \
    paths.base_results_dir="${PROJECT_STORAGE}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"
