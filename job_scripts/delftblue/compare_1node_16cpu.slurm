#!/bin/bash
#SBATCH --job-name=cmp_1n_16c
#SBATCH --partition=compute-p2
#SBATCH --account=research-ceg-gse
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_scripts/delftblue/out_files/slurm-cmp_1n_16c-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-cmp_1n_16c-%j.err

# Side A of the head-to-head: ONE node with 16 cpus, ensemble=32, ncpu=1.
# 32 single-rank uDALES members fight for 16 cpus on one node -> 2x OS
# oversubscription on the coordinator node. No srun fan-out (it's 1 node
# anyway, fan-out is moot). Pair with compare_2node_8cpu.slurm to compare
# wall time vs the same total cpu budget split across two nodes.

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

# Single-node allocation -> srun --exact fan-out has nowhere to fan out.
# Use the baseline (DISABLE_SRUN=1) path. Persist pool ON for steady-state
# fairness vs side B.
PYURBANAIR_DISABLE_SRUN=1 \
PYURBANAIR_PERSIST_POOL=1 \
pixi run -e delftblue -- python scripts/benchmark_ensemble_parallelism.py \
    --label "1node_16cpu" \
    --ensemble-size "${ENSEMBLE_SIZE}" \
    --num-parallel-processes "${NUM_PARALLEL}" \
    --ncpu "${NCPU}" \
    --num-calls "${NUM_CALLS}" \
    --simulation-time "${SIM_TIME}" \
    --spinup-time "${SPINUP_TIME}"

echo "Done on $(date)"
