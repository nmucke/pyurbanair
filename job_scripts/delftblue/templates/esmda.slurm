#!/bin/bash
# Generic time-varying-parameter rollout ESMDA template for DelftBlue. NOT meant
# to be `sbatch`ed directly — submit via the wrapper, which fills in
# --job-name, --partition, --cpus-per-task, --mem-per-cpu, --time,
# --output/--error and the PUA_* environment:
#
#     job_scripts/delftblue/submit.sh <assim_model> <size> [extra hydra overrides...]
#     TRUTH_MODEL=<m> job_scripts/delftblue/submit.sh <assim_model> <size> ...
#
# Truth and assimilation forward models are independent (PUA_TRUTH_MODEL /
# PUA_ASSIM_MODEL), so the same template covers single-model runs and twin
# experiments (e.g. truth=pyudales, assimilate with pylbm). Per-model solver
# flags + per-model environment workarounds are composed per role below.
#SBATCH --ntasks=1
#SBATCH --account=innovation

set -euo pipefail

# Run from the repo root regardless of where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

: "${PUA_SIZE:?PUA_SIZE not set — submit via job_scripts/delftblue/submit.sh}"
: "${PUA_NUM_PARALLEL:?PUA_NUM_PARALLEL not set — submit via job_scripts/delftblue/submit.sh}"
: "${PUA_TRUTH_MODEL:?PUA_TRUTH_MODEL not set — submit via job_scripts/delftblue/submit.sh}"
: "${PUA_ASSIM_MODEL:?PUA_ASSIM_MODEL not set — submit via job_scripts/delftblue/submit.sh}"

# Logs -> job_scripts/delftblue/out_files; temp/intermediate I/O -> scratch
# (/scratch/$USER, beegfs); final results -> ${PROJECT_STORAGE}.
PROJECT_STORAGE="/projects/urbanair"
SCRATCH_DIR="/scratch/${USER}"
RUN_TEMP_DIR="${SCRATCH_DIR}/urbanair_temp/${SLURM_JOB_ID}"
mkdir -p job_scripts/delftblue/out_files "${PROJECT_STORAGE}" "${RUN_TEMP_DIR}"

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Truth=${PUA_TRUTH_MODEL} assim=${PUA_ASSIM_MODEL} size=${PUA_SIZE} cores=${SLURM_CPUS_PER_TASK} num_parallel=${PUA_NUM_PARALLEL}"
echo "Results: ${PROJECT_STORAGE}  (temp: ${RUN_TEMP_DIR})"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# Force ob1/tcp so a stale pixi cache or unsourced activation can't pull
# UCX/InfiniBand into the MPI stack and crash MPI_Finalize on DelftBlue.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# Each ensemble worker pins to one core; keep in-rank BLAS single-threaded so
# workers x BLAS threads don't fight for the allocated cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Stderr is fully buffered when redirected to a file — without this a crash
# mid-run loses the traceback (cf. job 9842407).
export PYTHONUNBUFFERED=1

EXTRA_FLAGS=()

# Global: if pypalm is involved in either role, apply the PALM-specific
# environment + Hydra defaults that the dedicated pypalm scripts used to set.
case " ${PUA_TRUTH_MODEL} ${PUA_ASSIM_MODEL} " in
  *" pypalm "*)
    # PALM expects this vertical resolution (matches the historical pypalm scripts).
    EXTRA_FLAGS+=( "domain.nz=16" )

    # PALM launches a nested `mpirun palm` per ensemble member. The ensemble runs
    # each member in a forkserver worker pinned to a single core (default, tuned
    # for uDALES); confined to one CPU, OpenMPI's orted + palm rank with the
    # forced TCP BTL livelock and the member run stalls. Disable pinning so
    # members get the whole allocation, and let OMPI yield when idle / not bind
    # / oversubscribe so concurrent PALMs coexist.
    export PYURBANAIR_DISABLE_CPU_PINNING=1
    export OMPI_MCA_mpi_yield_when_idle=1
    export OMPI_MCA_rmaps_base_oversubscribe=true
    export OMPI_MCA_hwloc_base_binding_policy=none

    # Route the per-run PALM working dir (fast_io_catalog) to node-local /tmp
    # instead of beegfs scratch. With PYPALM_USE_DIRECT_RUN=1 (default below)
    # the stage is just a few symlinks + INPUT copies, so the choice of
    # filesystem barely matters; under the palmrun fallback path it's
    # load-bearing — palmrun copies the full build tree (~750 files) there per
    # member per window, and local SSD makes that cheap.
    export PYPALM_FAST_IO_CATALOG="/tmp/urbanair_palm_${SLURM_JOB_ID}"
    mkdir -p "${PYPALM_FAST_IO_CATALOG}"
    trap 'rm -rf "${PYPALM_FAST_IO_CATALOG}"' EXIT

    # M4: default to the direct-run path (no palmrun, no palmbuild). This
    # bypasses palmrun's ~130 s/invocation preamble and the mpirun-wrapped
    # combine, saving ~16x on tiny and ~2.4x on small, with bit-identical
    # u/v/w + posterior_params relative to the palmrun baseline. See
    # docs/palm_overhead_plan.md §M0–M3 for the validation trail.
    #
    # Escape hatch: prefix the submit with `PYPALM_USE_DIRECT_RUN=` to fall
    # back to palmrun, e.g.:
    #     PYPALM_USE_DIRECT_RUN= job_scripts/delftblue/submit.sh pypalm small
    # The forward model checks for the exact string "1"; any other value (or
    # unset) routes through the existing palmrun + execute.sh path.
    export PYPALM_USE_DIRECT_RUN="${PYPALM_USE_DIRECT_RUN:-1}"

    # The delftblue activation `module load`s nvhpc/25.7, which prepends NVHPC's
    # mpif90 (nvfortran) ahead of the conda env's gfortran wrapper. PALM's
    # installer then picks nvfortran, which can't read the gfortran-built
    # netcdf.mod and the build fails (cf. jobs 9949950 / 9950028). Drop any
    # stale CMakeCache.txt so it can't pin the wrong compiler; the inner bash
    # wrapper below then forces conda's bin (and unsets NVHPC's CC/... env) so
    # palmbuild + palmrun use the matching toolchain.
    if [ -d libs/pypalm/palm_model_system ]; then
        find libs/pypalm/palm_model_system -type f -name CMakeCache.txt -delete
    fi
    ;;
esac

# Global: pylbm's forward model hardcodes ./.temp relative to cwd. If either
# role is pylbm, point it at scratch via a symlink so intermediate I/O stays
# off the home quota; replace whatever is there so a stale link can't pin us back.
case " ${PUA_TRUTH_MODEL} ${PUA_ASSIM_MODEL} " in
  *" pylbm "*)
    rm -rf .temp
    ln -sfn "${RUN_TEMP_DIR}" .temp
    ;;
esac

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

while IFS= read -r flag; do
  [ -n "${flag}" ] && EXTRA_FLAGS+=( "${flag}" )
done < <(role_flags truth_model "${PUA_TRUTH_MODEL}"; role_flags assim_model "${PUA_ASSIM_MODEL}")

# Correlation localization is on by default (esmda.localization in the config).
# We pin a fixed truncation threshold so the run is well-posed at every ensemble
# size (the adaptive null default, 3/sqrt(N_e), excludes all obs for N_e < 9).
# Disable localization entirely with `esmda.localization=null`.
PY_ARGS=(
  python -u scripts/run_time_varying_parameters_rollout_esmda.py
  "size=${PUA_SIZE}"
  "model@truth_model=${PUA_TRUTH_MODEL}"
  "model@assim_model=${PUA_ASSIM_MODEL}"
  "ensemble.num_parallel_processes=${PUA_NUM_PARALLEL}"
  "esmda.localization.truncation_correlation=0.3"
  "paths.base_results_dir=${PROJECT_STORAGE}"
  "${EXTRA_FLAGS[@]}"
  "$@"
)

# If pypalm is in play, route through a bash -c that scrubs nvhpc's CC/F90/...
# and prepends CONDA_PREFIX/bin so palmbuild + palmrun pick conda's gfortran.
# Other models don't need this wrapping.
case " ${PUA_TRUTH_MODEL} ${PUA_ASSIM_MODEL} " in
  *" pypalm "*)
    pixi run -e delftblue -- bash -c '
      unset OPAL_PREFIX CC CXX F77 F90 FC
      export PATH="${CONDA_PREFIX}/bin:${PATH}"
      exec "$@"
    ' bash "${PY_ARGS[@]}"
    ;;
  *)
    pixi run -e delftblue -- "${PY_ARGS[@]}"
    ;;
esac
