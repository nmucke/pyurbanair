#!/usr/bin/env bash
# Thin wrapper around palmrun. Invoked by pypalm.forward_model.ForwardModel.run().
#
# Arguments:
#   $1  — experiment base directory (JOBS parent directory)
#   $2  — experiment name (palmrun -r <name>)
#   $3  — number of MPI tasks (palmrun -X)
#
# palmrun reads INPUT/<name>_p3d and writes OUTPUT/, MONITORING/, RESTART/
# under <base>/<name>/.
#
# Requires `palmrun` on PATH (set PALM_BIN, PALM_ROOT, or add to PATH before
# invoking the wrapper).
set -euo pipefail

base_dir="$1"
experiment_name="$2"
ncpu="$3"

if ! command -v palmrun >/dev/null 2>&1; then
    echo "execute.sh: palmrun not found on PATH. Set PALM_ROOT and source" 1>&2
    echo "             \$PALM_ROOT/.palm.config.<host>, or install palm_model_system." 1>&2
    exit 127
fi

cd "${base_dir}"

# palmrun flags used:
#   -r <name>  run identifier
#   -c default config identifier (matches what compile_palm wrote)
#   -a "d3#"   activation string: d3 = data_output 3d, # = interactive mode
#   -X <n>     number of MPI tasks
#   -T <n>     number of OpenMP threads (set equal to -X for single-node runs)
#   -q none    no queue submission
#   -v         verbose
exec palmrun \
    -r "${experiment_name}" \
    -c default \
    -a "d3#" \
    -X "${ncpu}" \
    -T "${ncpu}" \
    -q none \
    -v
