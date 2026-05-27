#!/bin/bash
#SBATCH --job-name=esmda_pylbm_small
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-esmda_pylbm_small-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-esmda_pylbm_small-%j.err

set -euo pipefail

# Run from the repo root regardless of where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

# Logs -> job_scripts/delftblue/out_files; temp/intermediate I/O -> scratch;
# final results -> ${PROJECT_STORAGE}.
PROJECT_STORAGE="/projects/urbanair"
SCRATCH_DIR="/scratch/${USER}"
RUN_TEMP_DIR="${SCRATCH_DIR}/urbanair_temp/${SLURM_JOB_ID}"
mkdir -p job_scripts/delftblue/out_files "${PROJECT_STORAGE}" "${RUN_TEMP_DIR}"

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Repo: $(pwd)"
echo "Results: ${PROJECT_STORAGE}  (temp: ${RUN_TEMP_DIR})"

module purge

# pylbm's forward model hardcodes ./.temp relative to cwd (i.e. inside this repo
# on /home). Point it at scratch via a symlink so intermediate I/O stays off the
# home/login quota; replace whatever is there so a stale link can't pin us back.
rm -rf .temp
ln -sfn "${RUN_TEMP_DIR}" .temp

# CPU run: the LBM CPU build uses gfortran directly (no nvfortran), and the
# assimilation math (JAX) stays on CPU on the compute partition (no GPU).
export JAX_PLATFORMS=cpu

# Each ensemble worker pins to one core; keep in-rank BLAS single-threaded so
# workers x BLAS threads don't fight for the allocated cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Stderr is fully buffered when redirected to a file — without this a crash
# mid-run loses the traceback (cf. job 9842407).
export PYTHONUNBUFFERED=1

pixi run -e delftblue -- python -u scripts/run_time_varying_parameters_rollout_esmda.py \
    size=small \
    model@truth_model=pylbm \
    model@assim_model=pylbm \
    truth_model.forward_model.cuda=false \
    assim_model.forward_model.cuda=false \
    ensemble.num_parallel_processes=8 \
    localization=correlation \
    paths.base_results_dir="${PROJECT_STORAGE}"
