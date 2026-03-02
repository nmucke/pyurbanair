#!/usr/bin/env bash
set -euo pipefail

# Downloads and installs NVIDIA HPC SDK (NVFORTRAN) outside conda.
# This is required because NVHPC is not available as a Pixi/conda package.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default to current public release tarball; override if needed.
# Example:
#   NVHPC_SDK_URL="https://developer.download.nvidia.com/hpc-sdk/24.9/nvhpc_2024_249_Linux_x86_64_cuda_multi.tar.gz" pixi run -e cuda install-nvhpc
NVHPC_SDK_URL="${NVHPC_SDK_URL:-https://developer.download.nvidia.com/hpc-sdk/26.1/nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz}"
DEFAULT_INSTALL_BASE="${PROJECT_ROOT}/.nvhpc"
if [ -n "${CONDA_PREFIX:-}" ]; then
  DEFAULT_INSTALL_BASE="${CONDA_PREFIX}/.nvhpc"
fi
NVHPC_INSTALL_BASE="${NVHPC_INSTALL_BASE:-${DEFAULT_INSTALL_BASE}}"
NVHPC_CACHE_DIR="${NVHPC_CACHE_DIR:-${PROJECT_ROOT}/.cache/nvhpc}"

if compgen -G "${NVHPC_INSTALL_BASE}/Linux_x86_64/*/compilers/bin/nvfortran" > /dev/null; then
  echo "NVHPC already installed in ${NVHPC_INSTALL_BASE}"
  exit 0
fi

mkdir -p "${NVHPC_CACHE_DIR}" "${NVHPC_INSTALL_BASE}"

ARCHIVE_NAME="$(basename "${NVHPC_SDK_URL}")"
ARCHIVE_PATH="${NVHPC_CACHE_DIR}/${ARCHIVE_NAME}"
EXTRACT_DIR="${NVHPC_CACHE_DIR}/extracted"

echo "Downloading NVHPC from ${NVHPC_SDK_URL}"
echo "This is a large download (~10-15 GB), so this may take a while."
curl -fL --continue-at - -o "${ARCHIVE_PATH}" "${NVHPC_SDK_URL}"

rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"

INSTALLER_DIR="$(ls -d "${EXTRACT_DIR}"/nvhpc_* | head -n 1)"
if [ -z "${INSTALLER_DIR}" ] || [ ! -x "${INSTALLER_DIR}/install" ]; then
  echo "Could not find NVHPC installer after extraction."
  exit 1
fi

echo "Installing NVHPC into ${NVHPC_INSTALL_BASE}"
NVHPC_SILENT=true \
NVHPC_INSTALL_TYPE=single \
NVHPC_INSTALL_DIR="${NVHPC_INSTALL_BASE}" \
"${INSTALLER_DIR}/install"

echo "NVHPC installation complete."
echo "Activate with: pixi shell --environment cuda"
