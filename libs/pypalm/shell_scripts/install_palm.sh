#!/usr/bin/env bash
# Install PALM (palm_model_system) against the active pixi environment.
#
# Arguments:
#   $1  —  Absolute path to an extracted palm_model_system source tree.
#          After successful install, ${1}/bin/palmrun and ${1}/bin/palmbuild
#          will exist.
#
# The install script autodetects compilers via nf-config / nc-config from
# $CONDA_PREFIX. We pass netcdf prefixes explicitly to avoid falling back to
# system paths when pixi deps are present.
# Don't use -e: `yes | bash install ...` naturally returns non-zero from the
# `yes` side of the pipe (SIGPIPE once install finishes). We check for the
# palmrun binary at the end instead.
set -uo pipefail

palm_root="$1"

if [ ! -d "${palm_root}" ]; then
    echo "install_palm.sh: palm_root not a directory: ${palm_root}" 1>&2
    exit 1
fi

if [ ! -f "${palm_root}/install" ]; then
    echo "install_palm.sh: install script missing under ${palm_root}" 1>&2
    exit 1
fi

cd "${palm_root}"

# Pick compiler / netcdf prefix from the pixi env when available.
prefix="${CONDA_PREFIX:-}"
compiler_flag=()
netcdf_c_flag=()
netcdf_fortran_flag=()

if command -v mpif90 >/dev/null 2>&1; then
    compiler_flag=(-c mpif90)
fi

if [ -n "${prefix}" ] && [ -d "${prefix}" ]; then
    netcdf_c_flag=(-s "${prefix}")
    netcdf_fortran_flag=(-t "${prefix}")
fi

echo "install_palm.sh: running install in ${palm_root}" 1>&2

# PALM's install asks y/n for optional components; pipe "yes" so we never block.
yes | bash install -p "${palm_root}" \
    "${compiler_flag[@]}" \
    "${netcdf_c_flag[@]}" \
    "${netcdf_fortran_flag[@]}"

# The `yes |` pipe above returns non-zero via SIGPIPE when install finishes,
# so PIPESTATUS[1] (install's own exit code) is what we care about.
install_exit="${PIPESTATUS[1]:-0}"

if [ ! -x "${palm_root}/bin/palmrun" ]; then
    echo "install_palm.sh: install exited ${install_exit} and bin/palmrun is missing." 1>&2
    exit 1
fi

echo "install_palm.sh: PALM installed at ${palm_root}/bin/palmrun" 1>&2
