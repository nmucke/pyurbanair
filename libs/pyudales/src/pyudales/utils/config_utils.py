"""Utilities for creating config.sh files for uDALES."""

import pathlib
import shlex

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
    executable = dirs.solver_executable or (
        dirs.udales_root_path / "build" / "release" / "u-dales"
    )
    values = {
        "DA_EXPDIR": dirs.experiment_base_dir,
        "DA_TOOLSDIR": dirs.udales_root_path / "tools",
        "DA_BUILD": executable,
        "DA_WORKDIR": dirs.output_dir,
        "NCPU": ncpu,
        "MATLAB_BIN": matlab_bin,
    }
    with open(config_sh_path, "w") as f:
        for key, value in values.items():
            f.write(f"export {key}={shlex.quote(str(value))}\n")
