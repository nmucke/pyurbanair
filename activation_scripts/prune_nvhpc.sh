#!/usr/bin/env bash
set -euo pipefail

# Trim an NVIDIA HPC SDK install down to the one CUDA version it links against.
#
# NVIDIA only ships the SDK as a "cuda_multi" tarball that bundles every
# supported CUDA toolkit (26.1: 12.9 and 13.1) side by side under math_libs/,
# comm_libs/, cuda/, profilers/ and REDIST/. nvfortran picks one of them at
# *compile* time from the host's GPU driver (localrc's DEFCUDAVERSION): a
# CUDA 12.x driver gets 12.9, an R580+ (CUDA 13) driver gets 13.1. The other
# toolkit(s) are ~12 GB of dead weight on that host.
#
# The version kept is the one nvfortran resolves on *this* host, so run the
# prune on a machine with the GPU driver the env will build on. Do not prune an
# install shared between hosts with different driver generations (a network
# filesystem, a copied env) -- set NVHPC_PRUNE=0 there. Without a working
# driver the choice cannot be detected and nothing is removed.
#
# Runs automatically at the end of install_nvhpc.sh; run it by hand to trim an
# existing install:
#     pixi run -e cuda prune-nvhpc
#
# Force the version kept with NVHPC_KEEP_CUDA=<ver>, or skip pruning entirely
# (e.g. to use Nsight Systems, which the SDK stores only under the newest
# toolkit's profilers/) with NVHPC_PRUNE=0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_INSTALL_BASE="${PROJECT_ROOT}/.nvhpc"
if [ -n "${CONDA_PREFIX:-}" ]; then
  DEFAULT_INSTALL_BASE="${CONDA_PREFIX}/.nvhpc"
fi
NVHPC_INSTALL_BASE="${NVHPC_INSTALL_BASE:-${DEFAULT_INSTALL_BASE}}"

if [ "${NVHPC_PRUNE:-1}" = "0" ]; then
  echo "NVHPC_PRUNE=0: keeping all bundled CUDA versions."
  exit 0
fi

shopt -s nullglob
for root in "${NVHPC_INSTALL_BASE}"/Linux_x86_64/*/; do
  root="${root%/}"
  # Skip the year alias (e.g. 2026 -> 26.1) so each release is visited once.
  [ -L "${root}" ] && continue
  [ -d "${root}/compilers" ] || continue

  keep="${NVHPC_KEEP_CUDA:-}"
  if [ -z "${keep}" ]; then
    if ! nvidia-smi >/dev/null 2>&1; then
      echo "No working GPU driver: cannot tell which CUDA version nvfortran will use; not pruning ${root}." >&2
      continue
    fi
    probe_dir="$(mktemp -d)"
    printf 'end\n' > "${probe_dir}/probe.f90"
    keep="$(cd "${probe_dir}" && "${root}/compilers/bin/nvfortran" -cuda -dryrun probe.f90 2>&1 \
      | grep -oE '/cuda/[0-9]+\.[0-9]+/' | head -n1 | cut -d/ -f3 || true)"
    rm -rf "${probe_dir}"
  fi
  if [ -z "${keep}" ] || [ ! -d "${root}/cuda/${keep}" ]; then
    echo "Could not determine the CUDA version nvfortran uses under ${root}; not pruning." >&2
    continue
  fi

  for component in math_libs comm_libs cuda profilers REDIST/math_libs REDIST/comm_libs REDIST/cuda; do
    for dir in "${root}/${component}"/[0-9]*.[0-9]*; do
      [ "$(basename "${dir}")" = "${keep}" ] && continue
      echo "Removing unused CUDA toolkit: ${dir}"
      rm -rf "${dir}"
    done
  done
  echo "Pruned ${root} to CUDA ${keep}."
done
