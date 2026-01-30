"""Utilities for creating config.sh files for uDALES."""

import os
import pathlib

from .dir_utils import DirectoryPaths


def create_config_sh(
    dirs: DirectoryPaths,
    matlab_bin: pathlib.Path,
    ncpu: int,
) -> None:
    """
    Create a config.sh file where the environment variables are set.

    DA_EXPDIR should point to the experiment_base_dir containing experiments,
    not the specific experiment directory, because MATLAB appends expnr to it.

    Args:
        dirs: DirectoryPaths instance containing experiment_base_dir, udales_root_path, and output_dir.
        matlab_bin: The path to the MATLAB binary.
        ncpu: The number of CPUs to use.
    """
    config_sh_path = dirs.experiment_dir / "config.sh"
    matlab_bin_dir = pathlib.Path(matlab_bin).parent
    # Set DA_EXPDIR to experiment_base_dir so MATLAB can append expnr

    udales_root_path = pathlib.Path(dirs.udales_root_path)
    da_expdir = dirs.experiment_base_dir
    with open(config_sh_path, "w") as f:
        f.write(f"export DA_EXPDIR={str(da_expdir)}\n")
        f.write(f"export DA_TOOLSDIR={str(udales_root_path.joinpath('tools'))}\n")
        f.write(
            f"export DA_BUILD={str(udales_root_path.joinpath('build', 'release', 'u-dales'))}\n"
        )
        f.write(f"export DA_WORKDIR={str(dirs.output_dir)}\n")
        f.write(f"export NCPU={ncpu}\n")
        f.write(f"export MATLAB_BIN={str(matlab_bin)}\n")
        # f.write(f"export PATH={matlab_bin_dir}:{os.environ.get('PATH', '')}\n")
        # Disable FFTW/OpenMP threading to prevent oversubscription when running
        # multiple MPI jobs in parallel. Each MPI job already uses multiple cores,
        # so additional FFTW threads cause severe cache thrashing and slowdowns.
        # f.write("export OMP_NUM_THREADS=1\n")
        # f.write("export FFTW_NUM_THREADS=1\n")
