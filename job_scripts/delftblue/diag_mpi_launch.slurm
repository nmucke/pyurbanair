#!/bin/bash
#SBATCH --job-name=diag_mpi_launch
#SBATCH --partition=compute-p2
#SBATCH --account=research-ceg-gse
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_scripts/delftblue/out_files/slurm-diag_mpi_launch-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-diag_mpi_launch-%j.err

# Diagnostic for "what mpiexec invocation actually works inside a 2-node
# allocation when we want each per-member mpiexec to be intra-node?"
# Each test runs `hostname` under mpiexec with -n 4 and a 30s timeout.
# Whichever path produces 4 lines of hostname output is a viable launch.

set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p job_scripts/delftblue/out_files

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "PATH=${PATH}"
echo "which srun: $(command -v srun || echo MISSING)"
echo

# Common env from the real benchmark
module purge
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0
export PRTE_MCA_plm=^slurm
export OMP_NUM_THREADS=1
export PYPALM_SKIP_AUTOINSTALL=1

# Make srun findable inside pixi env
if ! command -v srun >/dev/null 2>&1 \
   && [ -x /cm/shared/apps/slurm/current/bin/srun ]; then
    export PATH="/cm/shared/apps/slurm/current/bin:$PATH"
fi

run_test() {
    local label="$1"; shift
    echo "============================================================"
    echo "TEST: ${label}"
    echo "CMD : $*"
    echo "------------------------------------------------------------"
    timeout 30 "$@"
    local rc=$?
    echo "[${label}] exit=${rc}"
    echo
}

# -- Tests are ordered cheapest first; each runs through `pixi run` so
# -- the env matches what the real benchmark experiences.
PIXI="pixi run -e delftblue --"

run_test "T1_plain" \
    $PIXI mpiexec -n 4 hostname

run_test "T2_oversubscribe" \
    $PIXI mpiexec -n 4 --oversubscribe hostname

run_test "T3_host_localhost" \
    $PIXI mpiexec --host localhost:4 -n 4 hostname

run_test "T4_host_hostname" \
    $PIXI bash -c 'mpiexec --host $(hostname):4 -n 4 hostname'

run_test "T5_unset_slurm_nodelist" \
    $PIXI bash -c 'unset SLURM_JOB_NODELIST SLURM_NODELIST SLURM_NNODES \
                          SLURM_NPROCS SLURM_TASKS_PER_NODE SLURM_DISTRIBUTION; \
                   mpiexec -n 4 --oversubscribe hostname'

run_test "T6_prte_ras_simulator" \
    $PIXI bash -c 'PRTE_MCA_ras=simulator mpiexec -n 4 hostname' || true

run_test "T7_srun_exact_then_mpiexec_plain" \
    srun --exact -N1 -n1 --cpus-per-task=4 \
    $PIXI mpiexec -n 4 hostname

run_test "T8_srun_exact_then_unset_then_mpiexec" \
    srun --exact -N1 -n1 --cpus-per-task=4 \
    $PIXI bash -c 'unset SLURM_JOB_NODELIST SLURM_NODELIST SLURM_NNODES \
                          SLURM_NPROCS SLURM_TASKS_PER_NODE SLURM_DISTRIBUTION \
                          SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES; \
                   mpiexec -n 4 --oversubscribe hostname'

echo "Diagnostic complete on $(date)"
