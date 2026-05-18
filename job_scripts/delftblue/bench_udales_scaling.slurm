#!/bin/bash
#SBATCH --job-name=bench_udales
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --account=innovation
#SBATCH --output=job_scripts/delftblue/out_files/slurm-bench-%j.out
#SBATCH --error=job_scripts/delftblue/out_files/slurm-bench-%j.err

# Single-node uDALES ensemble parallel-scaling benchmark on DelftBlue.
# Sweeps num_parallel_processes (workers) at ncpu_per_member=1 across the
# allocated cores. Mirrors the local sweep in docs/ensemble_scaling.md so
# the resulting CSV is directly comparable.
#
# Output: .temp/bench/ensemble_scaling_delftblue_${SLURM_JOB_ID}.csv
#
# Recovery: the benchmark monkey-patches local_execute.sh in the pyudales
# install (to capture per-stage timings) and restores it in a finally
# block. The EXIT trap below restores it again as belt-and-suspenders if
# SLURM kills the job before Python's finally runs. If both fail, run:
#   pixi run -e delftblue -- python -c "import pathlib, shutil, pyudales.forward_model as fm; \
#       t=pathlib.Path(fm.LOCAL_EXECUTE_SCRIPT); b=t.with_suffix('.sh.bench-bak'); \
#       b.exists() and (shutil.copy(b,t), t.chmod(0o755), b.unlink())"

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p job_scripts/delftblue/out_files
mkdir -p .temp/bench

echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Repo: $(pwd)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Cores visible to this task: $(nproc)"
echo "CPU model: $(grep 'model name' /proc/cpuinfo | head -1 | sed 's/.*: //')"

module purge

# Same OMPI overrides as the production scripts: UCX/InfiniBand can crash
# in MPI_Finalize on DelftBlue, so force OB1 + TCP/shared-memory BTLs.
# Single-rank uDALES doesn't need RDMA.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0

# Each ensemble worker is a single-rank uDALES (ncpu=1). Force in-rank
# thread libraries to 1 so they don't fight the workers for cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

restore_pyudales() {
    pixi run -e delftblue -- python -c "
import pathlib, shutil
import pyudales.forward_model as fm
target = pathlib.Path(fm.LOCAL_EXECUTE_SCRIPT)
backup = target.with_suffix('.sh.bench-bak')
if backup.exists():
    shutil.copy(backup, target)
    target.chmod(0o755)
    backup.unlink()
    print(f'[trap] Restored {target}')
" 2>/dev/null || true
}
trap restore_pyudales EXIT

OUT_CSV=".temp/bench/ensemble_scaling_delftblue_${SLURM_JOB_ID}.csv"

pixi run -e delftblue -- python scripts/benchmark_ensemble_scaling.py \
    --ensemble-size 32 \
    --simulation-time 60.0 \
    --sweeps 1:1,1:4,1:8,1:16,1:32 \
    --out "${OUT_CSV}"

echo
echo "Bench complete. CSV: ${OUT_CSV}"
