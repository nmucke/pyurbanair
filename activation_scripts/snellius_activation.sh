#!/bin/bash
# Activation script for the Snellius supercomputer (CPU-only environment).
# This script is sourced by pixi during environment activation.
#
# Unlike the DelftBlue activation, this does NOT load a system NVHPC module:
# the Snellius `snellius` env is CPU-only and gets its full toolchain
# (gfortran, OpenMPI, FFTW3, netCDF) from conda-forge via pixi. We only need
# to (a) make conda's libs discoverable for linking/runtime and (b) tame the
# conda-forge OpenMPI shutdown crash on InfiniBand fabrics.

# Make pixi's own libraries (FFTW3, netCDF, ...) findable at link and run time.
# CONDA_PREFIX is set by pixi before this script is sourced.
if [ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    if [ -n "${LIBRARY_PATH}" ]; then
        export LIBRARY_PATH="${LIBRARY_PATH}:${CONDA_PREFIX}/lib"
    else
        export LIBRARY_PATH="${CONDA_PREFIX}/lib"
    fi

    if [ -n "${LD_LIBRARY_PATH}" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
    else
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
    fi
fi

# Conda-forge OpenMPI's UCX PML can segfault inside MPI_Finalize on
# InfiniBand fabrics (Snellius has IB): UCX detects the IB devices but can't
# fully initialize them, then crashes on shutdown after a clean run. Force the
# OB1 PML with TCP/shared-memory BTLs. The single-rank forward-model runs here
# don't need RDMA, and this works on both login and compute nodes.
#
# NB: this conda env ships OpenMPI 5.x, which renamed the shared-memory BTL
# `vader` -> `sm` and dropped the `pt2pt` one-sided (osc) component entirely.
# Do NOT set OMPI_MCA_osc=pt2pt here (as the DelftBlue/OpenMPI-4 activation
# does): forcing a non-existent osc component makes MPI_Init fail with
# "A requested component was not found" and aborts the uDALES solver. Leave osc
# unset so OpenMPI auto-selects (sm/rdma).
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,sm,tcp
export OMPI_MCA_btl_base_warn_component_unused=0
