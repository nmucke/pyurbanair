#!/bin/bash
#SBATCH --job-name=smoke_esmda_udales
#SBATCH --partition=compute-p2
#SBATCH --account=research-ceg-gse
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_scripts/delftblue/out_files/slurm-smoke_esmda_udales-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-smoke_esmda_udales-%j.err

# Tiny rollout-ESMDA smoke test (4 ensemble members, 1 window, 1 ESMDA step,
# 60 s sim time).  Peak compute demand: num_parallel_processes (4) *
# udales nprocx*nprocy (4) = 16 MPI ranks, plus headroom for the python
# coordinator / JAX, so --cpus-per-task=20 on one compute-p2 node.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

mkdir -p job_scripts/delftblue/out_files

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Repo:  $(pwd)"
echo "Nodes: ${SLURM_JOB_NODELIST:-?}  cpus-per-task=${SLURM_CPUS_PER_TASK:-?}"

module purge

# Belt-and-suspenders: even though delftblue_activation.sh sets these, set
# them again here so a stale pixi cache or an unsourced activation can't
# bring UCX/InfiniBand back into the MPI stack and crash MPI_Finalize.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# 16 udales MPI ranks share the slurm allocation; let the python
# coordinator and JAX use a single thread per worker so we don't
# oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# pyurbanair imports pypalm unconditionally (config_utils.py), and
# pypalm/__init__.py auto-installs PALM at import time when its compiled
# binary is missing. Compute nodes have no internet AND the kpp4palm
# subbuild hangs in a runaway recursive make on this tree, so skip the
# auto-install for udales-only runs.
export PYPALM_SKIP_AUTOINSTALL=1

pixi run -e delftblue -- python scripts/run_smoke_rollout_esmda.py \
    --truth-model pyudales --assim-model pyudales --skip-viz
