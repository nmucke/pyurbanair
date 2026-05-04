#!/bin/bash
# Activation script for LBM project
# Loads NVHPC module and sets up library paths for FFTW3
# This script is sourced by pixi during environment activation

# Debug: Show that script is being executed (uncomment to verify)
# echo "LBM activation script executing from: $(pwd)" >&2

# Load NVHPC module if available
if command -v module >/dev/null 2>&1; then
    echo "Loading NVHPC module"
    module load nvhpc/25.7 2>/dev/null || module load nvhpc 2>/dev/null || true
fi

# Append pixi lib directory to LIBRARY_PATH for FFTW3
# This allows linker to find FFTW3 while system MPI paths are searched first
# CONDA_PREFIX should be set by pixi before this script is sourced
if [ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    if [ -n "${LIBRARY_PATH}" ]; then
        export LIBRARY_PATH="${LIBRARY_PATH}:${CONDA_PREFIX}/lib"
    else
        export LIBRARY_PATH="${CONDA_PREFIX}/lib"
    fi

    # Also set LD_LIBRARY_PATH for runtime
    if [ -n "${LD_LIBRARY_PATH}" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
    else
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
    fi
fi

# Conda-forge OpenMPI's UCX PML segfaults inside MPI_Finalize on DelftBlue
# (after the run completes cleanly): UCX detects InfiniBand devices but
# can't fully initialize them — particularly on login nodes — then crashes
# on shutdown. Force the OB1 PML with TCP/shared-memory BTLs. Single-rank
# uDALES runs don't need RDMA, and this works on both login and compute nodes.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_btl_base_warn_component_unused=0
