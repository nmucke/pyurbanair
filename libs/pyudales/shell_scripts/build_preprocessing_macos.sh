#!/usr/bin/env bash
set -euo pipefail
source_dir="${1:-$(pwd)/u-dales}"
build_dir="${2:-$source_dir/tools/View3D/build}"
cmake -S "$source_dir/tools/View3D" -B "$build_dir" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$build_dir" --parallel "${UDALES_BUILD_JOBS:-4}"
