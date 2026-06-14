#!/usr/bin/env bash
#
# Submit a pyurbanair ESMDA job on DelftBlue, sizing the SLURM allocation from
# the experiment config instead of hard-coding cores per size.
#
# Usage:
#   job_scripts/delftblue/submit.sh <model> <size> [extra hydra overrides...]
#
#   <model>  pylbm | pyudales | pypalm   (the assimilation forward model; also
#            the truth model unless TRUTH_MODEL overrides it)
#   <size>   tiny | small | medium | large | xlarge   (a conf/size/<size>.yaml)
#
# The number of cores requested follows `ensemble.ensemble_size` from
# conf/size/<size>.yaml (one worker per ensemble member), capped at a single
# DelftBlue compute node (64 cores). For ensemble_size > 64 the wrapper
# oversubscribes workers up to 96 (matches the historical xlarge pattern). The
# matching worker count is passed as `ensemble.num_parallel_processes`. Edit
# ensemble_size in the size config and the allocation tracks it — no need to
# touch this script.
#
# Examples:
#   job_scripts/delftblue/submit.sh pylbm small
#   job_scripts/delftblue/submit.sh pyudales medium esmda.num_assimilation_windows=3
#   job_scripts/delftblue/submit.sh pypalm small ensemble.ensemble_size=20   # sizes for 20
#   TRUTH_MODEL=pyudales job_scripts/delftblue/submit.sh pylbm small   # twin: truth=pyudales, assim=pylbm
#
# Options (environment variables):
#   TRUTH_MODEL=<m>     forward model that generates the truth (default: <model>)
#   WALLTIME=HH:MM:SS   override the per-size default wall time
#   DRY_RUN=1           print the sbatch command and computed sizing, do not submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

die() { echo "error: $*" >&2; exit 1; }

(( $# >= 2 )) || die "usage: $(basename "$0") <model> <size> [extra hydra overrides...]"

ASSIM_MODEL="$1"; SIZE="$2"; shift 2   # remaining "$@" are extra hydra overrides
# Truth forward model defaults to the assimilation model; override with TRUTH_MODEL.
TRUTH_MODEL="${TRUTH_MODEL:-${ASSIM_MODEL}}"

for m in "${ASSIM_MODEL}" "${TRUTH_MODEL}"; do
  case "${m}" in
    pylbm|pyudales|pypalm) ;;
    *) die "unknown model '${m}' (expected pylbm|pyudales|pypalm)" ;;
  esac
done

TEMPLATE="${SCRIPT_DIR}/templates/esmda.slurm"
SIZE_CFG="${REPO_ROOT}/conf/size/${SIZE}.yaml"
[ -f "${TEMPLATE}" ] || die "missing template ${TEMPLATE}"
[ -f "${SIZE_CFG}" ] || die "no size config conf/size/${SIZE}.yaml (expected tiny|small|medium|large|xlarge)"

# Resolve ensemble_size: a CLI override (ensemble.ensemble_size=N) wins over the
# value in the size config, so the allocation matches what the run will use.
ENSEMBLE_SIZE=""
for arg in "$@"; do
  case "${arg}" in
    ensemble.ensemble_size=*) ENSEMBLE_SIZE="${arg#*=}" ;;
  esac
done
if [ -z "${ENSEMBLE_SIZE}" ]; then
  ENSEMBLE_SIZE="$(grep -E '^[[:space:]]*ensemble_size:' "${SIZE_CFG}" | head -1 | sed -E 's/.*:[[:space:]]*([0-9]+).*/\1/')"
fi
[[ "${ENSEMBLE_SIZE}" =~ ^[0-9]+$ ]] || die "could not read a numeric ensemble_size from ${SIZE_CFG} (got '${ENSEMBLE_SIZE}')"
(( ENSEMBLE_SIZE >= 1 )) || die "ensemble_size must be >= 1 (got ${ENSEMBLE_SIZE})"

# DelftBlue compute node sizing: cap cores at one compute-p2 node (64). pypalm
# tolerates oversubscription (it disables CPU pinning and lets OpenMPI yield),
# so workers can go up to NODE_OVERSUB (96) past the core cap — matches the
# historical xlarge pattern (96 workers on 64 cores). For pylbm/pyudales, one
# worker per core, no oversubscription.
NODE_MAX=64
NODE_OVERSUB=96

case " ${TRUTH_MODEL} ${ASSIM_MODEL} " in
  *" pypalm "*) WORKER_CAP=${NODE_OVERSUB} ;;
  *)            WORKER_CAP=${NODE_MAX} ;;
esac

NUM_PARALLEL=$(( ENSEMBLE_SIZE < WORKER_CAP ? ENSEMBLE_SIZE : WORKER_CAP ))
CORES=$(( NUM_PARALLEL < NODE_MAX ? NUM_PARALLEL : NODE_MAX ))

# Partition auto-selection: compute-p1 (48-core nodes, 218 of them) when the
# request fits, compute-p2 (64-core nodes, 90 of them) above that — the old
# combined `compute` partition is drained.
if (( CORES <= 48 )); then
  PARTITION="compute-p1"
else
  PARTITION="compute-p2"
fi

if (( ENSEMBLE_SIZE > WORKER_CAP )); then
  echo "warning: ensemble_size=${ENSEMBLE_SIZE} exceeds the worker cap (${WORKER_CAP});" >&2
  echo "         running ${NUM_PARALLEL} workers in parallel, the rest are processed in batches." >&2
fi

# pypalm worker processes are heavier on RSS (palm + combine subprocesses); the
# historical scripts allocated 2G/cpu for pypalm and 3G/cpu for the others. If
# either role is pypalm we keep the lower per-cpu memory; otherwise we use 3G.
case " ${TRUTH_MODEL} ${ASSIM_MODEL} " in
  *" pypalm "*) MEM_PER_CPU="2G" ;;
  *)            MEM_PER_CPU="3G" ;;
esac

# Per-size default wall time (override with WALLTIME=HH:MM:SS).
case "${SIZE}" in
  tiny)   DEF_TIME="00:30:00" ;;
  small)  DEF_TIME="04:00:00" ;;
  medium) DEF_TIME="12:00:00" ;;
  large)  DEF_TIME="24:00:00" ;;
  xlarge) DEF_TIME="24:00:00" ;;
  *)      DEF_TIME="12:00:00" ;;
esac
WALLTIME="${WALLTIME:-${DEF_TIME}}"

# Job name reflects the model(s): a single name when truth==assim, otherwise
# annotate the truth model so twin experiments are distinguishable in squeue.
if [ "${TRUTH_MODEL}" = "${ASSIM_MODEL}" ]; then
  JOBNAME="esmda_${ASSIM_MODEL}_${SIZE}"
else
  JOBNAME="esmda_${ASSIM_MODEL}_${SIZE}_truth-${TRUTH_MODEL}"
fi
OUT="job_scripts/delftblue/out_files/slurm-${JOBNAME}-%j.out"
ERR="job_scripts/delftblue/out_files/slurm-${JOBNAME}-%j.err"

echo "==> truth=${TRUTH_MODEL} assim=${ASSIM_MODEL} / ${SIZE}: ensemble_size=${ENSEMBLE_SIZE} -> partition=${PARTITION}, cores=${CORES}, num_parallel=${NUM_PARALLEL}, mem-per-cpu=${MEM_PER_CPU}, time=${WALLTIME}"
(( ${#} > 0 )) && echo "    extra hydra overrides: $*"

sbatch_cmd=(
  sbatch
  --job-name="${JOBNAME}"
  --partition="${PARTITION}"
  --cpus-per-task="${CORES}"
  --mem-per-cpu="${MEM_PER_CPU}"
  --time="${WALLTIME}"
  --output="${OUT}"
  --error="${ERR}"
  --chdir="${REPO_ROOT}"
  --export="ALL,PUA_SIZE=${SIZE},PUA_NUM_PARALLEL=${NUM_PARALLEL},PUA_TRUTH_MODEL=${TRUTH_MODEL},PUA_ASSIM_MODEL=${ASSIM_MODEL}"
  "${TEMPLATE}"
  "$@"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] ${sbatch_cmd[*]}"
  exit 0
fi

"${sbatch_cmd[@]}"
