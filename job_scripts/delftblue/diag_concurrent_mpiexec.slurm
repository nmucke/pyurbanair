#!/bin/bash
#SBATCH --job-name=diag_concurrent_mpi
#SBATCH --partition=compute-p2
#SBATCH --account=research-ceg-gse
#SBATCH --time=00:08:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_scripts/delftblue/out_files/slurm-diag_concurrent_mpi-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-diag_concurrent_mpi-%j.err

# Test the "N concurrent mpiexec invocations clash on PRTE session dir"
# hypothesis. T3 (`mpiexec --host localhost:N hostname`) was confirmed
# working in isolation by job 9840575, but hung silently when fired 20-way
# concurrently by the real benchmark (job 9839930).
#
# Updated for ncpu=1 (matches user's production usage; warmstart breaks at
# ncpu>1). Each member is now `mpiexec --host localhost:1 -n 1 hostname`
# (one rank per invocation). Concurrency sweep is 1 -> 4 -> 20 -> 80;
# 80 matches the real benchmark's `ensemble_size=80, num_parallel=80`.
# Tests both shared and per-invocation TMPDIR.

set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p job_scripts/delftblue/out_files

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo

module purge
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0
export PRTE_MCA_plm=^slurm
export OMP_NUM_THREADS=1
export PYPALM_SKIP_AUTOINSTALL=1

if ! command -v srun >/dev/null 2>&1 \
   && [ -x /cm/shared/apps/slurm/current/bin/srun ]; then
    export PATH="/cm/shared/apps/slurm/current/bin:$PATH"
fi

# Workspace for per-invocation logs and tmpdirs
WORK="${SLURM_SUBMIT_DIR}/.temp/diag_concurrent_${SLURM_JOB_ID}"
rm -rf "$WORK"
mkdir -p "$WORK"

# Run the actual concurrent test inside a single pixi activation, so all
# launched mpiexec invocations share the same env (matches the real
# benchmark, where one Python process activates pixi and forks 20 workers).
pixi run -e delftblue -- bash <<EOF
set -u

# --- shared functions -----------------------------------------------------
launch_one() {
    # \$1 = test label, \$2 = invocation index, \$3 = "shared" or "perinv" tmpdir mode
    local label="\$1" idx="\$2" mode="\$3"
    local out="${WORK}/\${label}_inv\${idx}.log"
    if [ "\$mode" = "perinv" ]; then
        local tmp="${WORK}/\${label}_inv\${idx}_tmp"
        mkdir -p "\$tmp"
        TMPDIR="\$tmp" timeout 60 mpiexec --host localhost:1 -n 1 hostname \\
            >"\$out" 2>&1
    else
        timeout 60 mpiexec --host localhost:1 -n 1 hostname \\
            >"\$out" 2>&1
    fi
    echo \$? > "${WORK}/\${label}_inv\${idx}.exitcode"
}
export -f launch_one
export WORK

# A test passes if all N invocations exited 0 with at least 1 hostname line.
run_test() {
    local label="\$1" n="\$2" mode="\$3"
    echo "============================================================"
    echo "TEST: \${label} (n=\${n}, tmpdir=\${mode})"
    echo "------------------------------------------------------------"
    local pids=()
    for i in \$(seq 1 "\$n"); do
        launch_one "\$label" "\$i" "\$mode" &
        pids+=(\$!)
    done
    for p in "\${pids[@]}"; do wait "\$p"; done

    local pass=0 fail=0
    for i in \$(seq 1 "\$n"); do
        local ec
        ec=\$(cat "${WORK}/\${label}_inv\${i}.exitcode" 2>/dev/null || echo "?")
        local lines
        lines=\$(wc -l < "${WORK}/\${label}_inv\${i}.log" 2>/dev/null || echo 0)
        if [ "\$ec" = "0" ] && [ "\$lines" -ge "1" ]; then
            pass=\$((pass+1))
        else
            fail=\$((fail+1))
            # surface first failure's tail to help diagnosis
            if [ "\$fail" -le 1 ]; then
                echo "  first failing invocation #\${i} (exit=\${ec}, lines=\${lines}):"
                tail -10 "${WORK}/\${label}_inv\${i}.log" 2>&1 | sed 's/^/    /'
            fi
        fi
    done
    echo "[\${label}] pass=\${pass}/\${n} fail=\${fail}"
    echo
}

# --- the actual sweep -----------------------------------------------------
# Concurrency progression maps to the real benchmark sizing.
run_test "T1_n1_shared"     1  shared
run_test "T2_n4_shared"     4  shared
run_test "T3_n20_shared"    20 shared
run_test "T4_n80_shared"    80 shared
run_test "T5_n80_perinv"    80 perinv

echo "Diagnostic complete on \$(date)"
EOF

echo
echo "All raw logs in: $WORK"
