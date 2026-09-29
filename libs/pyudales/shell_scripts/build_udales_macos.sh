#!/usr/bin/env bash
# Explicit source/build paths keep the pinned upstream checkout untouched.
set -euo pipefail
build_type="${1:-release}"
source_dir="${2:-$(pwd)/u-dales}"
build_dir="${3:-$source_dir/build/$build_type}"
case "$build_type" in
    debug|Debug) cmake_build_type=Debug ;;
    release|Release) cmake_build_type=Release ;;
    *) echo "Unsupported build type: $build_type" >&2; exit 1 ;;
esac
export FC="${FC:-mpif90}"
mkdir -p "$build_dir"
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DNETCDF_DIR="${NETCDF_DIR:-}" \
    -DNETCDF_FORTRAN_DIR="${NETCDF_FORTRAN_DIR:-}" \
    -DCMAKE_BUILD_TYPE="$cmake_build_type" \
    -DFFTW_DOUBLE_OPENMP_LIB="${FFTW_DOUBLE_LIB:-}" \
    -DFFTW_FLOAT_OPENMP_LIB="${FFTW_FLOAT_LIB:-}" \
    2>&1 | tee "$build_dir/config.log"
cmake --build "$build_dir" --parallel "${UDALES_BUILD_JOBS:-4}" \
    2>&1 | tee "$build_dir/build.log"
