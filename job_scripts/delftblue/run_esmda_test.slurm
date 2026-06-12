#!/bin/bash
# DELFTBLUE quick ESMDA test runner.
#
# Runs scripts/run_esmda.py with the CURRENT config setup, i.e. conf/run_esmda.yaml
# exactly as committed (smoother=dynamic, truth/assim=pyudales) with NO experiment
# overrides -- unlike the sweep runners this does not source common.sh. The only
# overrides are hydra.run.dir (outputs under test_outputs/), the per-job scratch
# temp_dir, and run.truth_dir (the loaded ground truth).
#
# Cores: the run uses ensemble.num_parallel_processes (16, conf/ensemble.yaml)
# workers at num_cpus_per_process=1, so 16 cores cover the concurrent need.
# Submit from the repo root:
#
#     sbatch job_scripts/delftblue/run_esmda_test.slurm
#     sbatch job_scripts/delftblue/run_esmda_test.slurm esmda.num_steps=2   # extra hydra overrides
#
# Point it at a different pre-simulated truth with TRUTH_DIR=<dir> (must hold
# state.nc + params.nc).
#SBATCH --job-name=run_esmda_test
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-run_esmda_test-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-run_esmda_test-%j.err

set -euo pipefail

# Run from the directory sbatch was invoked from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Pre-simulated ground truth to load (state.nc + params.nc).
TRUTH_DIR="${TRUTH_DIR:-/projects/urbanair/ground_truth}"

# Per-job output dir under test_outputs/ so repeated runs don't clobber each other.
RESULTS_DIR="${REPO_ROOT}/test_outputs/run_esmda_${SLURM_JOB_ID:-$$}"
# Heavy solver scratch goes to a per-job /scratch dir (beegfs).
RUN_TEMP_DIR="/scratch/${USER}/urbanair_temp/${SLURM_JOB_ID:-$$}"
mkdir -p job_scripts/delftblue/out_files "${RESULTS_DIR}" "${RUN_TEMP_DIR}"
# Clean this run's scratch on success; leave it for debugging on failure.
trap '[ "$?" = "0" ] && rm -rf "${RUN_TEMP_DIR}"' EXIT

# DelftBlue MPI environment: force ob1/tcp so a stale pixi cache or unsourced
# activation can't pull UCX/InfiniBand into the MPI stack and crash
# MPI_Finalize. Unlike Snellius, DelftBlue's OpenMPI DOES need osc=pt2pt.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# Keep BLAS single-threaded so the parallel ensemble workers (one per core) don't
# oversubscribe; line-buffer Python so a crash keeps its traceback.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "DELFTBLUE run_esmda test: job ${SLURM_JOB_ID:-local} on $(hostname) -- $(date)"
echo "Config: conf/run_esmda.yaml (current setup, no overrides)"
echo "Truth: ${TRUTH_DIR}"
echo "Cores=${SLURM_CPUS_PER_TASK:-?}  Output=${RESULTS_DIR}  Temp=${RUN_TEMP_DIR}"
[ "$#" -gt 0 ] && echo "Extra hydra overrides: $*"

module purge

# Run the current config as-is; redirect Hydra's run dir into test_outputs/ and
# the assim model's solver scratch into the per-job /scratch dir (the truth is
# loaded from disk, so only the assim model simulates). Any extra args ("$@") are
# appended as Hydra overrides and take precedence.
pixi run -e delftblue -- python -u \
    scripts/run_esmda.py \
    "hydra.run.dir=${RESULTS_DIR}" \
    "assim_model.forward_model.temp_dir=${RUN_TEMP_DIR}" \
    "run.truth_dir=${TRUTH_DIR}" \
    "run.ensemble_save_on_disk=true" \
    "$@"

echo "Done -- ESMDA outputs under ${RESULTS_DIR}"
