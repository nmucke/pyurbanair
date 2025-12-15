#!/usr/bin/env bash

set -e

# Usage: ./build_preprocessing_macos.sh [common / icl]

if [ ! -d u-dales ]; then
    echo "Please run this script from the pyudales directory (which should contain u-dales folder)"
    exit 1
fi

cd u-dales/tools/View3D
mkdir -p build
cd build

echo "Building View3D on local system."

# Use cmake from pixi environment (should be in PATH)
CMAKE_CMD=$(command -v cmake || which cmake || echo "cmake")
if [ ! -x "$CMAKE_CMD" ]; then
    echo "Error: cmake not found in PATH. Make sure pixi environment is activated." >&2
    exit 1
fi

$CMAKE_CMD -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
echo "View3D configuration complete."

make
