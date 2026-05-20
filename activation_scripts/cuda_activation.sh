#!/usr/bin/env bash

# Configure NVHPC (nvfortran) compiler paths for the cuda Pixi environment.
#
# This script is sourced on EVERY activation, so it must be fast and must
# never block. It only exports paths when a complete NVHPC install is already
# present. If NVHPC is missing it prints a one-line hint and returns success.
#
# Installation is a separate, explicit, one-time step:
#     pixi run -e cuda install-nvhpc
#
# Keeping the (~16 GB) download/install out of activation means an interrupted
# install can never masquerade as an activation "hang", and a broken partial
# install is never produced by simply activating the environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_INSTALL_BASE="${PROJECT_ROOT}/.nvhpc"
if [ -n "${CONDA_PREFIX:-}" ]; then
  DEFAULT_INSTALL_BASE="${CONDA_PREFIX}/.nvhpc"
fi
NVHPC_INSTALL_BASE="${NVHPC_INSTALL_BASE:-${DEFAULT_INSTALL_BASE}}"

NVFORTRAN_BIN=""
for candidate in "${NVHPC_INSTALL_BASE}"/Linux_x86_64/*/compilers/bin; do
  if [ -x "${candidate}/nvfortran" ]; then
    NVFORTRAN_BIN="${candidate}"
  fi
done

if [ -z "${NVFORTRAN_BIN}" ]; then
  echo "[cuda env] NVHPC (nvfortran) is not installed under ${NVHPC_INSTALL_BASE}."
  echo "[cuda env] Enable GPU builds with a one-time install:  pixi run -e cuda install-nvhpc"
  # Sourced during activation: return; tolerate direct execution too.
  return 0 2>/dev/null || exit 0
fi

NVHPC_VERSION_DIR="$(cd "${NVFORTRAN_BIN}/.." && pwd)"
NVHPC_ROOT="$(cd "${NVHPC_VERSION_DIR}/.." && pwd)"

export NVCOMPILERS="${NVHPC_INSTALL_BASE}"
export PATH="${NVFORTRAN_BIN}:${PATH}"

if [ -d "${NVHPC_ROOT}/comm_libs/mpi/bin" ]; then
  export PATH="${NVHPC_ROOT}/comm_libs/mpi/bin:${PATH}"
fi

if [ -d "${NVHPC_VERSION_DIR}/lib" ]; then
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${NVHPC_VERSION_DIR}/lib:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${NVHPC_VERSION_DIR}/lib"
  fi
fi
