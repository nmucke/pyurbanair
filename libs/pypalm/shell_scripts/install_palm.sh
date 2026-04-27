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

# The perl `palmrun` wrapper ships in the source tarball and exists as soon
# as the source is extracted, so it isn't proof of a successful build. The
# actual Fortran binary at MAKE_DEPOSITORY_default/palm and the source tar
# at MAKE_DEPOSITORY_default/palm_current_version.tar are what palmrun needs
# to compile per-run cases. Gate on those instead.
palm_bin="${palm_root}/MAKE_DEPOSITORY_default/palm"
palm_src_tar="${palm_root}/MAKE_DEPOSITORY_default/palm_current_version.tar"
if [ ! -x "${palm_bin}" ] || [ ! -f "${palm_src_tar}" ]; then
    echo "install_palm.sh: install exited ${install_exit};" 1>&2
    echo "                 missing $(basename "${palm_bin}") and/or $(basename "${palm_src_tar}")" 1>&2
    echo "                 under ${palm_root}/MAKE_DEPOSITORY_default/." 1>&2
    exit 1
fi

# macOS-only fixup: rrtmg's Makefile links the .so without -install_name, so
# rrtmg.so ends up with LC_ID_DYLIB = "rrtmg.so" (bare). The linker bakes
# that bare name into palm's LC_LOAD_DYLIB. At runtime palm runs from a
# fresh tmp dir (palmrun extracts MAKE_DEPOSITORY_default's tar there), so
# dyld can't find rrtmg.so. We can't fall back to DYLD_LIBRARY_PATH because
# /bin/bash and openmpi's mpirun are SIP-protected and strip DYLD_* across
# exec, even with mpirun's `-x` forwarding.
#
# Instead, rewrite both load commands to the absolute path. palmrun copies
# (does not rebuild) palm + rrtmg.so out of palm_current_version.tar for
# each run unless USER_CODE is present, so we also have to re-pack the tar
# after patching, otherwise per-run extraction reverts to the bare-name
# binaries.
if [ "$(uname)" = "Darwin" ]; then
    rrtmg_so="${palm_root}/MAKE_DEPOSITORY_default/rrtmg/rrtmg.so"
    palm_tar="${palm_root}/MAKE_DEPOSITORY_default/palm_current_version.tar"
    if [ -f "${rrtmg_so}" ] && command -v install_name_tool >/dev/null 2>&1; then
        install_name_tool -id "${rrtmg_so}" "${rrtmg_so}"
        install_name_tool -change "rrtmg.so" "${rrtmg_so}" "${palm_bin}"
        if [ -f "${palm_tar}" ]; then
            (cd "${palm_root}/MAKE_DEPOSITORY_default" \
             && tar -uf "$(basename "${palm_tar}")" palm rrtmg/rrtmg.so)
        fi
    fi
fi

echo "install_palm.sh: PALM installed at ${palm_bin}" 1>&2
