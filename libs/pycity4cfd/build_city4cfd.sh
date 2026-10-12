#!/usr/bin/env bash
# Compile City4CFD from the pinned submodule into libs/pycity4cfd/build/.
#
# Run through `pixi run build-city4cfd` (dev or mcp env): every dependency comes
# from the pixi env (CMAKE_PREFIX_PATH), the submodule is never written to, and
# re-running only rebuilds what changed. The binary is build/city4cfd.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/City4CFD"
build="$here/build"

if [ ! -f "$src/CMakeLists.txt" ]; then
    git -C "$here" submodule update --init City4CFD
fi

case "$(uname -s)" in
    # Conda's clang with the Xcode command line tools' SDK. -B/usr/bin/ links
    # with Apple's ld: conda's ld64 cannot read current SDKs (see
    # pyurbanair.utils.toolchain.apple_linker_flags).
    Darwin)
        compilers=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
                   -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)"
                   -DCMAKE_EXE_LINKER_FLAGS=-B/usr/bin/
                   -DCMAKE_SHARED_LINKER_FLAGS=-B/usr/bin/
                   -DCMAKE_FIND_FRAMEWORK=NEVER) ;;
    *)
        compilers=(-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++) ;;
esac

cmake -S "$src" -B "$build" "${compilers[@]}" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target city4cfd -j "$(getconf _NPROCESSORS_ONLN)"
echo "City4CFD built: $build/city4cfd"
