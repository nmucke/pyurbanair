#!/usr/bin/env bash

# Configure NVHPC compiler paths if locally installed.

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

if [ -n "${NVFORTRAN_BIN}" ]; then
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
else
  echo "NVHPC not installed in ${NVHPC_INSTALL_BASE}; installing now..."
  if ! bash "${SCRIPT_DIR}/install_nvhpc.sh"; then
    echo "NVHPC installation failed during CUDA activation."
    return 1
  fi

  NVFORTRAN_BIN=""
  for candidate in "${NVHPC_INSTALL_BASE}"/Linux_x86_64/*/compilers/bin; do
    if [ -x "${candidate}/nvfortran" ]; then
      NVFORTRAN_BIN="${candidate}"
    fi
  done

  if [ -n "${NVFORTRAN_BIN}" ]; then
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
  else
    echo "NVHPC was installed but nvfortran was not found."
    return 1
  fi
fi
