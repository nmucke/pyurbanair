#!/bin/bash
#SBATCH --job-name=bench_parallelism
#SBATCH --partition=compute-p2
#SBATCH --account=research-ceg-gse
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_scripts/delftblue/out_files/slurm-bench_parallelism-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-bench_parallelism-%j.err

# Benchmark four parallelism modes back-to-back, on the SAME slurm
# allocation, so the only thing that varies is the env-var gating in our
# code:
#
#   baseline   PYURBANAIR_DISABLE_SRUN=1, PYURBANAIR_PERSIST_POOL=0
#              -> mpiexec --oversubscribe, fresh pool per call.
#              All 24 mpiexec * 4 ranks = 96 cores fight for node 0's
#              64 cores -> 1.5x oversubscribed.
#
#   A only     srun fan-out, fresh pool per call.
#              24 srun --exact slots distributed across 2 nodes
#              (~12/node = 48 cores/node) -> no oversubscription.
#
#   C only    PYURBANAIR_DISABLE_SRUN=1, persistent pool.
#              Pool created once across calls -> no re-fork cost.
#              Still oversubscribed on node 0.
#
#   A+C        srun fan-out + persistent pool. Both wins together.
#
# Allocation: 2 compute-p2 nodes, 16 task slots * 4 cpus = 64 task-cpus
# per node = 128 cores total. ensemble_size=24 needs 24*4=96 task-cpus,
# so srun easily fits across the allocation.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

mkdir -p job_scripts/delftblue/out_files

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Repo:       $(pwd)"
echo "Nodes:      ${SLURM_JOB_NODELIST:-?}"
echo "tasks/node: ${SLURM_NTASKS_PER_NODE:-?}  cpus/task: ${SLURM_CPUS_PER_TASK:-?}"

module purge

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0
# Disable PRTE's slurm PLM: each per-member mpiexec is intra-node, so we
# want PRTE to fork ranks locally instead of routing through srun.
# (Without this, mpiexec inside the bash subprocess auto-detects SLURM_*
# env vars and tries to invoke srun — which is not on the subprocess PATH
# AND would also defeat the per-member -N1 placement we set up via
# `srun --exact` in local_execute.sh.)
# NOTE: must be `PRTE_MCA_*` for OpenMPI 5 / PRTE — `OMPI_MCA_plm` is
# silently ignored by the runtime layer. local_execute.sh re-exports this
# at its own scope as belt-and-suspenders against pixi env stripping.
export PRTE_MCA_plm=^slurm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export PYPALM_SKIP_AUTOINSTALL=1

# Sizing rationale
# ----------------
# Each member runs single-threaded (ncpu=1) — matches the user's production
# usage (warmstart does not work for ncpu>1). With ncpu=1, change A's
# per-node oversubscription signal requires `ensemble * ncpu > 64` (single
# compute-p2 node has 64 cpus); pick 80 for ~1.25x oversubscription on a
# single node, mirrors the prior `ensemble=20, ncpu=4` design.
#
#   ensemble_size 80 * ncpu 1 = 80 task-cpus needed.
#   single-node compute-p2 has 64 cpus -> 1.25x oversubscribed (baseline).
#   2-node allocation has 128 task-cpus -> A's srun fan-out fits cleanly.
#
# Per-member wall is shorter than the ncpu=4 design (no MPI overhead per
# member, smaller working set), so the larger ensemble fits in the same
# 20-min slurm wall:
#   sim_time 3 s, num_calls 2 -> ~1 cold + 1 warm per scenario.
#   4 scenarios * (cold + warm) ~= 12-16 min compute.
# `steady_mean` is averaged over 1 warm call (noisier than the original 2);
# the speedup signal is still dominated by sim time vs noise floor.
ENSEMBLE_SIZE=80
NUM_PARALLEL=80
NUM_CALLS=2
SIM_TIME=3
SPINUP_TIME=2
NCPU=1

run_scenario() {
    local label="$1"
    local disable_srun="$2"
    local persist="$3"
    echo
    echo "============================================================"
    echo "  scenario: ${label}"
    echo "  PYURBANAIR_DISABLE_SRUN=${disable_srun}"
    echo "  PYURBANAIR_PERSIST_POOL=${persist}"
    echo "============================================================"
    PYURBANAIR_DISABLE_SRUN="${disable_srun}" \
    PYURBANAIR_PERSIST_POOL="${persist}" \
    pixi run -e delftblue -- python scripts/benchmark_ensemble_parallelism.py \
        --label "${label}" \
        --ensemble-size "${ENSEMBLE_SIZE}" \
        --num-parallel-processes "${NUM_PARALLEL}" \
        --ncpu "${NCPU}" \
        --num-calls "${NUM_CALLS}" \
        --simulation-time "${SIM_TIME}" \
        --spinup-time "${SPINUP_TIME}"
}

# Order: baseline first so the cache is cold for the slowest scenario;
# A+C last so the steady-state win is unambiguous.
run_scenario "baseline_no_srun_no_persist" "1" "0"
run_scenario "C_only_no_srun_persist"      "1" "1"
run_scenario "A_only_srun_no_persist"      ""  "0"
run_scenario "A_plus_C_srun_persist"       ""  "1"

echo
echo "Benchmark complete on $(date)"
