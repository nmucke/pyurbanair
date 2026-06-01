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

# Global: if pypalm is involved in either role, apply the PALM-specific
# environment + Hydra defaults. Mirrors the DelftBlue template, minus the
# DelftBlue-only nvhpc/CMakeCache scrub (Snellius's pixi env doesn't load
# nvhpc) and the OMPI_MCA=osc=pt2pt override (OpenMPI 5 on Snellius removed
# that component — setting it crashes MPI_Init; see memory).
case " ${PUA_TRUTH_MODEL} ${PUA_ASSIM_MODEL} " in
  *" pypalm "*)
    # PALM expects this vertical resolution (matches the DelftBlue pypalm scripts).
    EXTRA_FLAGS+=( "domain.nz=16" )

    # PALM launches a nested `mpirun palm` per ensemble member. Workers pinned
    # to a single core deadlock with the nested MPI ranks; disable pinning and
    # let OMPI yield/no-bind/oversubscribe so concurrent PALMs coexist.
    export PYURBANAIR_DISABLE_CPU_PINNING=1
    export OMPI_MCA_mpi_yield_when_idle=1
    export OMPI_MCA_rmaps_base_oversubscribe=true
    export OMPI_MCA_hwloc_base_binding_policy=none

    # Per-run PALM working dir on node-local scratch ($TMPDIR =
    # /scratch-local/<jobid> on Snellius). Auto-purged at job end, so no
    # cleanup trap needed. direct_palm copies outputs to OUTPUT/ before
    # tearing the tempdir down.
    export PYPALM_FAST_IO_CATALOG="${TMPDIR}/urbanair_palm_${SLURM_JOB_ID}"
    mkdir -p "${PYPALM_FAST_IO_CATALOG}"

    # Default to direct-run (no palmrun, no palmbuild): ~16x faster on tiny,
    # ~2.4x on small, bit-identical u/v/w + posterior_params relative to the
    # palmrun baseline. See docs/palm_overhead_plan.md §M0–M3. Escape hatch:
    # prefix the submit with `PYPALM_USE_DIRECT_RUN=` to fall back to palmrun.
    # NB: direct-run reuses the prebuilt binary at
    # libs/pypalm/palm_model_system/MAKE_DEPOSITORY_default/{palm,combine_plot_fields.x};
    # if it's absent the run aborts. Compute nodes have no internet, so build
    # it once on a login node before submitting.
    export PYPALM_USE_DIRECT_RUN="${PYPALM_USE_DIRECT_RUN:-1}"
    ;;
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

# Correlation localization is on by default (esmda.localization in the config).
# We pin a fixed truncation threshold so the run is well-posed at every ensemble
# size (the adaptive null default, 3/sqrt(N_e), excludes all obs for N_e < 9).
# Disable localization entirely with `esmda.localization=null`.
pixi run -e snellius -- python -u scripts/run_time_varying_parameters_rollout_esmda.py \
    size="${PUA_SIZE}" \
    model@truth_model="${PUA_TRUTH_MODEL}" \
    model@assim_model="${PUA_ASSIM_MODEL}" \
    ensemble.num_parallel_processes="${PUA_NUM_PARALLEL}" \
    esmda.localization.truncation_correlation=0.3 \
    paths.base_results_dir="${PROJECT_STORAGE}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"
