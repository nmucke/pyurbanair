#!/usr/bin/env bash
#
# Submit a pyurbanair ESMDA job on Snellius, sizing the SLURM allocation from
# the experiment config instead of hard-coding cores per size.
#
# Usage:
#   job_scripts/snellius/submit.sh <model> <size> [extra hydra overrides...]
#
#   <model>  pylbm | pyudales | pypalm   (the assimilation forward model; also
#            the truth model unless TRUTH_MODEL overrides it)
#   <size>   tiny | small | medium | large | xlarge   (a conf/size/<size>.yaml)
#
# The number of cores requested follows `ensemble.ensemble_size` from
# conf/size/<size>.yaml (one worker per ensemble member), rounded up to the
# partition's minimum billable share (16 on rome, 24 on genoa) and capped at a
# single node. The matching worker count is passed through as
# `ensemble.num_parallel_processes`. Edit ensemble_size in the size config and
# the allocation tracks it automatically — no need to touch this script.
#
# Examples:
#   job_scripts/snellius/submit.sh pylbm small
#   job_scripts/snellius/submit.sh pyudales medium esmda.num_assimilation_windows=3
#   job_scripts/snellius/submit.sh pypalm small ensemble.ensemble_size=20   # sizes for 20
#   TRUTH_MODEL=pyudales job_scripts/snellius/submit.sh pylbm small   # twin: truth=pyudales, assim=pylbm
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

# Pick partition + sizing. rome (128 cores) handles up to a full node; beyond
# that, genoa (192). Min billable share is node/8: 16 on rome, 24 on genoa.
if (( ENSEMBLE_SIZE <= 128 )); then
  PARTITION="rome";  NODE_MAX=128; GRAN=16
else
  PARTITION="genoa"; NODE_MAX=192; GRAN=24
fi

NUM_PARALLEL=$(( ENSEMBLE_SIZE < NODE_MAX ? ENSEMBLE_SIZE : NODE_MAX ))
CORES=$(( (NUM_PARALLEL + GRAN - 1) / GRAN * GRAN ))   # round up to billing share
(( CORES <= NODE_MAX )) || CORES=${NODE_MAX}

if (( ENSEMBLE_SIZE > NODE_MAX )); then
  echo "warning: ensemble_size=${ENSEMBLE_SIZE} exceeds a single ${PARTITION} node (${NODE_MAX} cores);" >&2
  echo "         running ${NUM_PARALLEL} workers in parallel, the rest are processed in batches." >&2
fi

# Per-size default wall time (override with WALLTIME=HH:MM:SS).
case "${SIZE}" in
  tiny)   DEF_TIME="00:30:00" ;;
  small)  DEF_TIME="16:00:00" ;;
  medium) DEF_TIME="24:00:00" ;;
  large)  DEF_TIME="48:00:00" ;;
  xlarge) DEF_TIME="96:00:00" ;;
  *)      DEF_TIME="24:00:00" ;;
esac
WALLTIME="${WALLTIME:-${DEF_TIME}}"

# Job name reflects the model(s): a single name when truth==assim, otherwise
# annotate the truth model so twin experiments are distinguishable in squeue.
if [ "${TRUTH_MODEL}" = "${ASSIM_MODEL}" ]; then
  JOBNAME="${ASSIM_MODEL}_${SIZE}"
else
  JOBNAME="${ASSIM_MODEL}_${SIZE}_truth-${TRUTH_MODEL}"
fi
OUT="job_scripts/snellius/out_files/slurm-${JOBNAME}-%j.out"
ERR="job_scripts/snellius/out_files/slurm-${JOBNAME}-%j.err"

# Per-submission working directory. Concurrent jobs that share the repo root
# as SLURM_SUBMIT_DIR clobber each other's mutable state — most notably the
# `.temp` symlink pylbm uses (the template does `rm -rf .temp && ln -sfn
# $RUN_TEMP_DIR .temp`, and $RUN_TEMP_DIR is a node-local /scratch-local path
# only valid on the submitting node). The fix: each submission gets its own
# `--chdir`, populated with symlinks back to the read-only parts of the repo
# (source, configs, pixi env, examples, libs source + cached builds). The
# template's `.temp` then lives inside that per-job dir and can't conflict with
# any other in-flight job. Workdirs are tiny (just symlinks); /scratch-shared
# auto-purges them — no explicit cleanup needed.
JOB_WORKDIR_BASE="/scratch-shared/${USER}/urbanair_runs"
JOB_WORKDIR="${JOB_WORKDIR_BASE}/$(date +%Y%m%d-%H%M%S)-${JOBNAME}-$$"
# Dry-run doesn't submit anything, so don't litter scratch with a workdir we
# won't use — just print the path that would be created.
if [ "${DRY_RUN:-0}" != "1" ]; then
  mkdir -p "${JOB_WORKDIR}"
  for item in \
      pyproject.toml \
      pixi.lock \
      conda-pypi-mapping.json \
      .pixi \
      src \
      scripts \
      conf \
      libs \
      activation_scripts \
      examples \
      job_scripts \
  ; do
    [ -e "${REPO_ROOT}/${item}" ] && ln -s "${REPO_ROOT}/${item}" "${JOB_WORKDIR}/${item}"
  done
fi

echo "==> truth=${TRUTH_MODEL} assim=${ASSIM_MODEL} / ${SIZE}: ensemble_size=${ENSEMBLE_SIZE} -> partition=${PARTITION}, cores=${CORES}, num_parallel=${NUM_PARALLEL}, time=${WALLTIME}"
echo "    workdir: ${JOB_WORKDIR}"
(( ${#} > 0 )) && echo "    extra hydra overrides: $*"

sbatch_cmd=(
  sbatch
  --job-name="${JOBNAME}"
  --partition="${PARTITION}"
  --cpus-per-task="${CORES}"
  --time="${WALLTIME}"
  --output="${OUT}"
  --error="${ERR}"
  --chdir="${JOB_WORKDIR}"
  --export="ALL,PUA_SIZE=${SIZE},PUA_NUM_PARALLEL=${NUM_PARALLEL},PUA_TRUTH_MODEL=${TRUTH_MODEL},PUA_ASSIM_MODEL=${ASSIM_MODEL}"
  "${TEMPLATE}"
  "$@"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] ${sbatch_cmd[*]}"
  exit 0
fi

"${sbatch_cmd[@]}"
