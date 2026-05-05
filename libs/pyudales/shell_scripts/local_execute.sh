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

## On DelftBlue (and other slurm clusters) pixi env activation strips the
## slurm binary directory from PATH, so `srun` and other slurm tools are
## not findable from inside the python subprocess that invokes us. Restore
## the standard slurm bin dir so (a) the `command -v srun` check below
## actually fires, and (b) any tools that introspect via srun work.
if [ -n "${SLURM_JOB_ID:-}" ] \
   && ! command -v srun >/dev/null 2>&1 \
   && [ -x /cm/shared/apps/slurm/current/bin/srun ]; then
    export PATH="/cm/shared/apps/slurm/current/bin:$PATH"
fi

## Pixi also strips SLURM_CONF, which srun needs to find slurmctld. Without
## it srun falls back to DNS SRV lookup (`_slurmctld._tcp` records), which
## DelftBlue doesn't publish — failing with `DNS SRV lookup failed`.
## Restore the canonical DelftBlue path if missing.
if [ -n "${SLURM_JOB_ID:-}" ] \
   && [ -z "${SLURM_CONF:-}" ] \
   && [ -r /cm/shared/apps/slurm/var/etc/delftblue/slurm.conf ]; then
    export SLURM_CONF=/cm/shared/apps/slurm/var/etc/delftblue/slurm.conf
fi

## OpenMPI 5 / PRTE auto-detects `SLURM_*` env vars and routes mpiexec
## launches through the slurm PLM (which calls srun). For our per-member
## intra-node mpiexec we want PRTE to fork ranks locally instead — exclude
## slurm from the PLM candidate set so PRTE falls back to its ssh PLM
## (which short-circuits to local fork for single-host launches).
## NOTE: this is `PRTE_MCA_*` (OpenMPI 5 runtime) not `OMPI_MCA_*`
## (which only controls the MPI layer and is silently ignored by PRTE).
export PRTE_MCA_plm=^slurm

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

## Per-member TMPDIR so concurrent mpiexec invocations from a Python
## ProcessPool don't collide on PRTE's session directory
## (`$TMPDIR/prte.<user>.<jobid>`). Without isolation, N simultaneous
## mpiexec launches inside the same SLURM_JOB_ID stomp on each other's
## sockets/PMIx state and silently wedge — observed empirically when the
## working `mpiexec --host localhost:N` invocation hung at 20-way
## concurrency in the real benchmark while running fine in isolation.
export TMPDIR="$outdir/.prte_tmp"
mkdir -p "$TMPDIR"

## execute program with mpi
##
## `--host localhost:$NCPU` is essential inside a slurm allocation: without
## it, PRTE auto-detects the full SLURM_JOB_NODELIST and tries to spread
## the $NCPU ranks across all allocated nodes, which then needs slurm-PLM
## (calls srun) or ssh-PLM (needs passwordless ssh between compute nodes)
## to launch the remote half — neither is available here. Pinning to
## localhost makes it a single-host launch that PRTE handles via local
## fork. The per-member intra-node placement is the right behavior anyway:
## the warm-start file is per-member.
##
## Inside a slurm allocation, wrap with `srun --exact -N1 -n1` so
## concurrent ensemble members spread across the allocation (single node
## when --nodes=1, multi-node when --nodes>1) instead of all crowding onto
## the coordinator's node and oversubscribing. The outer srun moves the
## bash subprocess to a free node; the inner mpiexec then launches the
## $NCPU ranks locally on that node. Wrapping mpiexec rather than letting
## srun launch the ranks directly avoids slurm-MPI PMI handshake issues
## with the conda-forge OpenMPI shipped via pixi.
##
## Set PYURBANAIR_DISABLE_SRUN=1 to fall back to plain mpiexec
## --oversubscribe (used by the parallelism benchmark to measure the
## baseline behavior).
if [ -n "${SLURM_JOB_ID:-}" ] \
   && [ -z "${PYURBANAIR_DISABLE_SRUN:-}" ] \
   && command -v srun >/dev/null 2>&1; then
    srun --exact -N1 -n1 --cpus-per-task=$NCPU \
        mpiexec --host localhost:$NCPU -n $NCPU $DA_BUILD namoptions.$exp 2>&1 \
        | tee -a run.$exp.log
else
    mpiexec --host localhost:$NCPU -n $NCPU --oversubscribe $DA_BUILD namoptions.$exp 2>&1 \
        | tee -a run.$exp.log
fi

## Merge output files across outputs.
## Always run gather_outputs.sh to merge per-processor files
## (even with NCPU=1, uDALES writes files with processor indices)
echo "Merging outputs across cores into one..."
$DA_TOOLSDIR/gather_outputs.sh $outdir

popd

echo "Simulation for case $exp ran sucesfully!"
