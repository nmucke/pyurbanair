#!/bin/bash
#SBATCH --job-name=cmp_2n_8c
#SBATCH --partition=compute-p2
#SBATCH --account=research-ceg-gse
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_scripts/delftblue/out_files/slurm-cmp_2n_8c-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-cmp_2n_8c-%j.err

# Side B of the head-to-head: TWO nodes with 8 cpus each = 16 cpus total
# (matches side A's total budget), ensemble=32, ncpu=1. Uses change A's
# `srun --exact` fan-out so members spread across both nodes instead of
# crowding the coordinator.
#
# Slurm task accounting: 8 task-slots/node * 2 nodes = 16 slots total,
# each `srun --exact -N1 -n1 --cpus-per-task=1` consumes 1 slot, so 16
# of the 32 ensemble members run concurrently (8 per node, no per-node
# oversub); the other 16 wait for slots to free, giving 2 sequential
# batches per call. This is the multi-node analogue of side A's 1-batch
# 2x-OS-oversub run -- both ultimately push 32 single-rank simulations
# through 16 cpus, just with the contention placed differently:
#   side A: oversub at the OS scheduler level on one node
#   side B: oversub at the slurm task accounting level, work split nodes
# The interesting question is which placement is faster for this workload.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p job_scripts/delftblue/out_files

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Nodes:      ${SLURM_JOB_NODELIST:-?}"
echo "tasks/node: ${SLURM_NTASKS_PER_NODE:-?}  cpus/task: ${SLURM_CPUS_PER_TASK:-?}"
echo "Total cpus: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))"
echo

module purge
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0
export PRTE_MCA_plm=^slurm
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYPALM_SKIP_AUTOINSTALL=1

# Sizing
ENSEMBLE_SIZE=32
NUM_PARALLEL=32
NUM_CALLS=2
SIM_TIME=3
SPINUP_TIME=2
NCPU=1

# Multi-node allocation -> srun --exact fan-out enabled (DISABLE_SRUN unset).
# Persist pool ON for steady-state fairness vs side A.
PYURBANAIR_PERSIST_POOL=1 \
pixi run -e delftblue -- python scripts/benchmark_ensemble_parallelism.py \
    --label "2node_8cpu" \
    --ensemble-size "${ENSEMBLE_SIZE}" \
    --num-parallel-processes "${NUM_PARALLEL}" \
    --ncpu "${NCPU}" \
    --num-calls "${NUM_CALLS}" \
    --simulation-time "${SIM_TIME}" \
    --spinup-time "${SPINUP_TIME}"

echo "Done on $(date)"
