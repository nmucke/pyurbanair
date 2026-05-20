#!/usr/bin/env bash
set -euo pipefail

# Downloads, extracts, and installs the NVIDIA HPC SDK (provides nvfortran)
# into the active Pixi environment. NVHPC is not packaged for conda/Pixi, so
# it lives under ${CONDA_PREFIX}/.nvhpc.
#
# Run it explicitly (it is NOT triggered by environment activation):
#     pixi run -e cuda install-nvhpc
#
# The script is idempotent and self-verifying:
#   * exits early if a usable nvfortran is already installed;
#   * verifies the downloaded archive (gzip -t) before extracting;
#   * always extracts into a clean directory; and
#   * fails loudly if nvfortran is missing after install, so an interrupted
#     download/extraction can never masquerade as a working SDK.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default to current public release tarball; override if needed, e.g.:
#   NVHPC_SDK_URL="https://developer.download.nvidia.com/hpc-sdk/24.9/nvhpc_2024_249_Linux_x86_64_cuda_multi.tar.gz" pixi run -e cuda install-nvhpc
NVHPC_SDK_URL="${NVHPC_SDK_URL:-https://developer.download.nvidia.com/hpc-sdk/26.1/nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz}"
DEFAULT_INSTALL_BASE="${PROJECT_ROOT}/.nvhpc"
if [ -n "${CONDA_PREFIX:-}" ]; then
  DEFAULT_INSTALL_BASE="${CONDA_PREFIX}/.nvhpc"
fi
NVHPC_INSTALL_BASE="${NVHPC_INSTALL_BASE:-${DEFAULT_INSTALL_BASE}}"
NVHPC_CACHE_DIR="${NVHPC_CACHE_DIR:-${PROJECT_ROOT}/.cache/nvhpc}"

# Echo the path to an installed, executable nvfortran (empty if none).
nvfortran_path() {
  local hit
  hit="$(compgen -G "${NVHPC_INSTALL_BASE}/Linux_x86_64/*/compilers/bin/nvfortran" 2>/dev/null | head -n1 || true)"
  if [ -n "${hit}" ] && [ -x "${hit}" ]; then
    printf '%s' "${hit}"
  fi
}

if [ -n "$(nvfortran_path)" ]; then
  echo "NVHPC already installed: $(nvfortran_path)"
  exit 0
fi

mkdir -p "${NVHPC_CACHE_DIR}"

ARCHIVE_NAME="$(basename "${NVHPC_SDK_URL}")"
ARCHIVE_PATH="${NVHPC_CACHE_DIR}/${ARCHIVE_NAME}"
EXTRACT_DIR="${NVHPC_CACHE_DIR}/extracted"

echo "Downloading NVHPC from ${NVHPC_SDK_URL}"
echo "This is a large download (~16 GB) and may take a while (resumable)."
curl -fL --retry 3 --retry-delay 5 --continue-at - -o "${ARCHIVE_PATH}" "${NVHPC_SDK_URL}"

echo "Verifying archive integrity..."
if ! gzip -t "${ARCHIVE_PATH}"; then
  echo "Downloaded archive is corrupt or truncated: ${ARCHIVE_PATH}" >&2
  echo "Delete it and re-run to download afresh." >&2
  exit 1
fi

echo "Extracting into a clean ${EXTRACT_DIR} ..."
rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"

INSTALLER_DIR="$(ls -d "${EXTRACT_DIR}"/nvhpc_* 2>/dev/null | head -n 1 || true)"
if [ -z "${INSTALLER_DIR}" ] || [ ! -x "${INSTALLER_DIR}/install" ]; then
  echo "Could not find NVHPC installer after extraction." >&2
  exit 1
fi

echo "Installing NVHPC into ${NVHPC_INSTALL_BASE} (this also takes a while)..."
rm -rf "${NVHPC_INSTALL_BASE}"
mkdir -p "${NVHPC_INSTALL_BASE}"
NVHPC_SILENT=true \
NVHPC_INSTALL_TYPE=single \
NVHPC_INSTALL_DIR="${NVHPC_INSTALL_BASE}" \
"${INSTALLER_DIR}/install"

if [ -z "$(nvfortran_path)" ]; then
  echo "NVHPC install finished but nvfortran is still missing under" >&2
  echo "${NVHPC_INSTALL_BASE} — the extraction or install was incomplete." >&2
  echo "Check free disk space and re-run 'pixi run -e cuda install-nvhpc'." >&2
  exit 1
fi

echo "NVHPC installation complete: $(nvfortran_path)"
echo "The cuda environment will now find nvfortran automatically on activation."
