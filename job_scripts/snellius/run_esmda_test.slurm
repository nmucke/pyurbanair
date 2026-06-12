#!/bin/bash
# SNELLIUS quick ESMDA test runner.
#
# Runs scripts/run_esmda.py with the CURRENT config setup, i.e. conf/run_esmda.yaml
# exactly as committed (smoother=dynamic, truth/assim=pyudales) with NO experiment
# overrides -- unlike the sweep runners this does not source common.sh. The only
# overrides are hydra.run.dir (outputs under test_outputs/), the per-job scratch
# temp_dir, and run.truth_dir (the loaded ground truth).
#
# Cores: the run uses ensemble.num_parallel_processes (16) workers at
# num_cpus_per_process=1, so 16 cores are the concurrent need -- which is also
# Snellius' 16-core billing minimum. Submit from the repo root:
#
#     sbatch job_scripts/snellius/run_esmda_test.slurm
#     sbatch job_scripts/snellius/run_esmda_test.slurm esmda.num_steps=2   # extra hydra overrides
#
#SBATCH --job-name=run_esmda_test
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=1750M
#SBATCH --account=tdsr72361
#SBATCH --output=job_scripts/snellius/out_files/slurm-run_esmda_test-%j.out
#SBATCH --error=job_scripts/snellius/out_files/slurm-run_esmda_test-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Per-job output dir under test_outputs/ so repeated runs don't clobber each other.
RESULTS_DIR="${REPO_ROOT}/test_outputs/run_esmda_${SLURM_JOB_ID:-$$}"
# Heavy solver scratch goes to a per-job /scratch-shared dir (NOT /scratch-local --
# pyudales same-model runs fail there; see job_scripts/snellius/README.md).
RUN_TEMP_DIR="/scratch-shared/${USER}/urbanair_temp/${SLURM_JOB_ID:-$$}"
mkdir -p job_scripts/snellius/out_files "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

# Keep BLAS single-threaded so the parallel ensemble workers (one per core) don't
# oversubscribe; line-buffer Python so a crash keeps its traceback.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "SNELLIUS run_esmda test: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Config: conf/run_esmda.yaml (current setup, no overrides)"
echo "Cores=${SLURM_CPUS_PER_TASK:-?}  Output=${RESULTS_DIR}  Temp=${RUN_TEMP_DIR}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# Run the current config as-is; redirect Hydra's run dir into test_outputs/ and the
# assim model's solver scratch into the per-job /scratch-shared dir (the truth is
# loaded from disk, so only the assim model simulates). Any extra args ("$@") are
# appended as Hydra overrides and take precedence.
pixi run -e snellius -- python -u \
    scripts/run_esmda.py \
    "hydra.run.dir=${RESULTS_DIR}" \
    "assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}" \
    "run.truth_dir=/projects/prjs2075/urbanair/ground_truth_small_small/pyudales_time_varying" \
    "run.ensemble_save_on_disk=true" \
    "$@"

echo "Done -- ESMDA outputs under ${RESULTS_DIR}"
