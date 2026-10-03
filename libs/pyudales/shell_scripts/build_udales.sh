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
# NetCDF and FFTW come from the pixi env. Left to CMake's own search, a system
# NetCDF without the Fortran module or a system FFTW (through the system
# pkg-config) can win. Values the caller sets still take precedence.
prefix="${CONDA_PREFIX:?build uDALES inside the pixi env (pixi run / pixi shell)}"
export FC="${FC:-mpif90}"
mkdir -p "$build_dir"
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH="$prefix" \
    -DNETCDF_DIR="${NETCDF_DIR:-$prefix}" \
    -DNETCDF_FORTRAN_DIR="${NETCDF_FORTRAN_DIR:-$prefix}" \
    -DFFTW_ROOT="${FFTW_ROOT:-$prefix}" \
    -DCMAKE_BUILD_TYPE="$cmake_build_type" \
    -DFFTW_DOUBLE_OPENMP_LIB="${FFTW_DOUBLE_LIB:-}" \
    -DFFTW_FLOAT_OPENMP_LIB="${FFTW_FLOAT_LIB:-}" \
    2>&1 | tee "$build_dir/config.log"
cmake --build "$build_dir" --parallel "${UDALES_BUILD_JOBS:-4}" \
    2>&1 | tee "$build_dir/build.log"
