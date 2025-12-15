#!/usr/bin/env bash

# uDALES (https://github.com/uDALES/u-dales).

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Copyright (C) 2016-2019 the uDALES Team.

set -e

# Usage: ./tools/build_executable.sh [icl, archer, cca, common] [debug, release]

if [ ! -d u-dales ]; then
    echo "Please run this script from the pyudales directory (which should contain u-dales folder)"
    exit 1
fi

normalize_build_type() {
    case "$1" in
        debug|Debug)
            echo "Debug"
            ;;
        release|Release)
            echo "Release"
            ;;
        *)
            echo "Unsupported build type '$1'. Use 'debug' or 'release'." >&2
            exit 1
            ;;
    esac
}

#echo "--- Debug info ---"
#echo "env: " `env`
#echo "PATH: " ${PATH}

NPROC=4 # TODO: make into a arg var.
build_type=$1

# Use mpif90 from pixi environment if FC is not set
if [ -z "$FC" ]; then
    if command -v mpif90 >/dev/null 2>&1; then
        FC=mpif90
    else
        echo "Error: No working Fortran compiler (mpif90) found in pixi environment" >&2
        exit 1
    fi
fi

# Configure and Build
path_to_build_dir="$(pwd)/u-dales/build/$build_type"
mkdir -p $path_to_build_dir
pushd $path_to_build_dir
cmake_build_type="$(normalize_build_type $build_type)"

# Use cmake from pixi environment (should be in PATH)
CMAKE_CMD=$(command -v cmake || which cmake || echo "cmake")
if [ ! -x "$CMAKE_CMD" ]; then
    echo "Error: cmake not found in PATH. Make sure pixi environment is activated." >&2
    exit 1
fi

# Patch all CMakeLists files to fix minimum version requirement
UDALES_ROOT="../../"
BACKUP_FILES=()

# Function to patch a CMake file
patch_cmake_file() {
    local file="$1"
    if [ -f "$file" ]; then
        # Check if file has old version (< 3.5)
        if grep -q "cmake_minimum_required(VERSION [0-2]\." "$file" 2>/dev/null || \
           grep -q "cmake_minimum_required(VERSION 3\.[0-4]" "$file" 2>/dev/null; then
            # Backup original
            cp "$file" "${file}.bak"
            BACKUP_FILES+=("${file}.bak")
            # Update minimum version to 3.5 (macOS compatible sed)
            # Match patterns like: cmake_minimum_required(VERSION 2.8.2) or cmake_minimum_required(VERSION 3.4)
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' 's/cmake_minimum_required(VERSION [0-3]\.[0-9][0-9.]*)/cmake_minimum_required(VERSION 3.5)/g' "$file"
            else
                sed -i 's/cmake_minimum_required(VERSION [0-3]\.[0-9][0-9.]*)/cmake_minimum_required(VERSION 3.5)/g' "$file"
            fi
            echo "Patched $file" >&2
        fi
    fi
}

# Patch main CMakeLists.txt and downloadFindFFTW.cmake.in
patch_cmake_file "${UDALES_ROOT}CMakeLists.txt"
patch_cmake_file "${UDALES_ROOT}downloadFindFFTW.cmake.in"

FC=$FC $CMAKE_CMD -DNETCDF_DIR=$NETCDF_DIR \
                  -DNETCDF_FORTRAN_DIR=$NETCDF_FORTRAN_DIR \
                  -DCMAKE_BUILD_TYPE=$cmake_build_type \
                  -DFFTW_DOUBLE_OPENMP_LIB=$FFTW_DOUBLE_LIB \
                  -DFFTW_FLOAT_OPENMP_LIB=$FFTW_FLOAT_LIB \
                  ../.. 2>&1 | tee -a $path_to_build_dir/config.log
CMAKE_EXIT_CODE=${PIPESTATUS[0]}

# Restore original files from backups
for backup in "${BACKUP_FILES[@]}"; do
    if [ -f "$backup" ]; then
        original="${backup%.bak}"
        mv "$backup" "$original"
    fi
done

# Exit if cmake failed
if [ $CMAKE_EXIT_CODE -ne 0 ]; then
    echo "CMake configuration failed with exit code $CMAKE_EXIT_CODE" >&2
    exit $CMAKE_EXIT_CODE
fi

make -j$NPROC 2>&1 | tee -a $path_to_build_dir/build.log
popd
