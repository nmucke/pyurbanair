#!/usr/bin/env bash
set -euo pipefail

# Build netcdf-fortran with NVFORTRAN so netcdf.mod is compatible with CUDA builds.
# Idempotent: skips if a working install already exists.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "CONDA_PREFIX is not set. Run this from an active pixi environment."
  exit 1
fi

NETCDF_C_PREFIX="${NETCDF_C_PREFIX:-${CONDA_PREFIX}}"
NVHPC_NETCDF_PREFIX="${NVHPC_NETCDF_PREFIX:-${CONDA_PREFIX}/.nvhpc/netcdf-fortran}"
NETCDF_FORTRAN_VERSION="${NETCDF_FORTRAN_VERSION:-4.6.2}"
NETCDF_FORTRAN_TARBALL="v${NETCDF_FORTRAN_VERSION}.tar.gz"
NETCDF_FORTRAN_URL="${NETCDF_FORTRAN_URL:-https://github.com/Unidata/netcdf-fortran/archive/refs/tags/${NETCDF_FORTRAN_TARBALL}}"
CACHE_DIR="${NETCDF_FORTRAN_CACHE_DIR:-${PROJECT_ROOT}/.cache/netcdf-fortran}"
SRC_DIR="${CACHE_DIR}/netcdf-fortran-${NETCDF_FORTRAN_VERSION}"
BUILD_DIR="${CACHE_DIR}/build-${NETCDF_FORTRAN_VERSION}"
ARCHIVE_PATH="${CACHE_DIR}/netcdf-fortran-${NETCDF_FORTRAN_TARBALL}"

if { [ -f "${NVHPC_NETCDF_PREFIX}/lib/libnetcdff.so" ] || [ -f "${NVHPC_NETCDF_PREFIX}/lib/libnetcdff.a" ]; } \
  && [ -f "${NVHPC_NETCDF_PREFIX}/include/netcdf.mod" ]; then
  echo "NVHPC-compatible netcdf-fortran already installed at ${NVHPC_NETCDF_PREFIX}"
  exit 0
fi

if ! command -v nvfortran >/dev/null 2>&1; then
  echo "nvfortran not found in PATH. Activate CUDA env first (pixi shell -e cuda)."
  exit 1
fi

mkdir -p "${CACHE_DIR}" "${NVHPC_NETCDF_PREFIX}"

if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "Downloading netcdf-fortran ${NETCDF_FORTRAN_VERSION} from ${NETCDF_FORTRAN_URL}"
  curl -fL --retry 3 --retry-delay 3 -o "${ARCHIVE_PATH}" "${NETCDF_FORTRAN_URL}"
fi

rm -rf "${SRC_DIR}" "${BUILD_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${CACHE_DIR}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${NVHPC_NETCDF_PREFIX}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_Fortran_COMPILER=nvfortran \
  -DCMAKE_C_COMPILER="${CC:-gcc}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF \
  -DENABLE_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DENABLE_DOXYGEN=OFF \
  -DNETCDF_C_LIBRARY="${NETCDF_C_PREFIX}/lib/libnetcdf.so" \
  -DNETCDF_C_INCLUDE_DIR="${NETCDF_C_PREFIX}/include" \
  -DCMAKE_PREFIX_PATH="${NETCDF_C_PREFIX}"

cmake --build "${BUILD_DIR}" --parallel
cmake --install "${BUILD_DIR}"

if [ ! -f "${NVHPC_NETCDF_PREFIX}/include/netcdf.mod" ]; then
  echo "Failed to install netcdf.mod to ${NVHPC_NETCDF_PREFIX}/include"
  exit 1
fi
if [ ! -f "${NVHPC_NETCDF_PREFIX}/lib/libnetcdff.so" ] && [ ! -f "${NVHPC_NETCDF_PREFIX}/lib/libnetcdff.a" ]; then
  echo "Failed to install netcdf-fortran to ${NVHPC_NETCDF_PREFIX}"
  exit 1
fi

echo "Installed NVHPC-compatible netcdf-fortran to ${NVHPC_NETCDF_PREFIX}"
