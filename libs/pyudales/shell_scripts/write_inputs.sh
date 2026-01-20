#!/usr/bin/env bash

# uDALES (https://github.com/uDALES/u-dales).
# Python-based preprocessing script

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
# Modified for Python implementation

# script for setting key namoptions and writing *.inp.* files
# This Python version replaces the MATLAB-based write_inputs.m

set -e

if (( $# < 1 )) 
then
    echo "The path to case/experiment folder must be set."
	echo "usage: FROM THE TOP LEVEL DIRECTORY run: libs/pyudales/shell_scripts/write_inputs.sh <PATH_TO_CASE>"
	echo "... execution terminated"
    exit 1
fi

start=${2:-"x"}     # pass 'c' if needs to be run on hpc compute node, or 'l' if to be run on login node

# go to experiment directory
pushd $1
	inputdir=$(pwd)

	## set experiment number via path
	iexpnr="${inputdir: -3}"

	## read in additional variables
	if [ -f config.sh ]; then
   	 source config.sh
	else
	 echo "config.sh must be set inside $inputdir"
     exit 1
	fi

	## check if required variables are set
	if [ -z $DA_TOOLSDIR ]; then
	    echo "Script directory DA_TOOLSDIR must be set inside $inputdir/config.sh"
	    exit 1
	fi;
	if [ -z $DA_EXPDIR ]; then
		echo "Experiment directory DA_EXPDIR must be set $inputdir/config.sh"
		exit 1
	fi;

popd

if [ $start == "c" ]; then

	cd $inputdir

###### RUN PYTHON SCRIPT through HPC job script
cat <<EOF > pre-job.$iexpnr
#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=64gb

module load tools/prod
module load Python/3.12
module load GCC/14.2.0

cd $DA_EXPDIR

export DA_TOOLSDIR=$DA_TOOLSDIR
export DA_EXPDIR=$DA_EXPDIR

python -m pyudales.python_udgeom.write_inputs $iexpnr > $inputdir/write_inputs.$iexpnr.log 2>&1

EOF

## submit job.exp file to queue
	qsub pre-job.$iexpnr
	echo "pre-job.$iexpnr submitted."
else
	###### RUN PYTHON SCRIPT
	cd $DA_EXPDIR
	# Run Python preprocessing script
	# Note: Ensure pyudales package is installed or PYTHONPATH includes the package
	python -m pyudales.python_udgeom.write_inputs $iexpnr > $inputdir/write_inputs.$iexpnr.log 2>&1
	cd ..
fi

