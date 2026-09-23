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

# Usage: ./tools/local_execute.sh <PATH_TO_CASE>

set -e
set -o pipefail

if (( $# < 1 ))
then
    echo "The path to case folder must be set."
    exit
fi

## go to experiment directory
pushd $1
inputdir=$(pwd)

## set experiment number via path
exp="${inputdir: -3}"

echo "Setting up uDALES for case $exp..."

## read in additional variables
if [ -f config.sh ]; then
    source config.sh
fi

## check if required variables are set
## or set default if not
if [ -z $NCPU ]; then
    NCPU=1
fi;
if [ -z $DA_WORKDIR ]; then
    echo "Output top-level directory DA_WORKDIR must be set"
    exit
fi;
if [ -z $DA_BUILD ]; then
    echo "Executable DA_BUILD must be set"
    exit
fi;
if [ -z $DA_TOOLSDIR ]; then
    echo "Script directory DA_TOOLSDIR must be set"
    exit
fi;

## set the experiment output directory
outdir=$DA_WORKDIR/$exp

echo "Starting job for case $exp..."

## copy files to output directory
mkdir -p $outdir
cp -r ./* $outdir

## go to execution and output directory
pushd $outdir

## execute program with mpi
##
## --bind-to none: respect the parent process affinity (set by
##   pyurbanair.utils.cpu_pinning when running ensembles), so MPI does
##   not move ranks off the CPUs the worker was pinned to.
## --oversubscribe: still allow more ranks than OpenMPI's slot count
##   (cgroups/affinity can confuse the slot computation under pinning).
## Restrict UCX to shared memory + self.
##
## This host has a DOWN network interface that reports speed = -1
## (/sys/class/net/enp6s0/speed). UCX divides by the port speed in
## ucp_worker_iface_port_speed (core/ucp_worker.c:812) while enumerating
## transports, so EVERY rank takes a SIGFPE during MPI init -- the run dies
## with exit 136 before a single timestep, and the Fortran backtrace looks
## like a solver crash even though no uDALES code has run yet.
##
## This script launches mpiexec with no hostfile, so it is single-node by
## construction and shared memory is all the transport it needs (it is also
## the fastest option for ranks on one box). Only set when the caller has not
## chosen a transport list themselves.
if [ -z "$UCX_TLS" ]; then
    export UCX_TLS=sm,self
fi

mpiexec -n $NCPU --bind-to none --oversubscribe $DA_BUILD namoptions.$exp 2>&1 | tee -a run.$exp.log

## Merge output files across outputs.
## Always run gather_outputs.sh to merge per-processor files
## (even with NCPU=1, uDALES writes files with processor indices)
echo "Merging outputs across cores into one..."
$DA_TOOLSDIR/gather_outputs.sh $outdir

popd

echo "Simulation for case $exp ran sucesfully!"
