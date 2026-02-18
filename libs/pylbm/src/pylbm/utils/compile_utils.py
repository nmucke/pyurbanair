"""
Compile the LBM program.
"""

import logging
import os
import pathlib
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

from .dir_utils import DirectoryPaths
from .makefile_utils import Makefile


def compile_lbm(
    dirs: DirectoryPaths,
    verbose: bool = True,
    enable_netcdf: bool = True,
) -> None:
    """
    Compile the LBM program.

    This function:
    1. Updates the HOME path in the makefile to the pixi environment path
    2. Changes to the LBM src directory
    3. Runs make to compile the program

    Args:
        dirs: DirectoryPaths object containing all relevant paths (including lbm_src_path,
              makefile_path, and pixi_env_path).
        verbose: If True, print compilation output. If False, suppress output.
        enable_netcdf: If True, enable NETCDF compilation flag.

    Raises:
        FileNotFoundError: If makefile or lbm_src_path doesn't exist.
        RuntimeError: If compilation fails.
    """
    if not dirs.makefile_path.exists():
        raise FileNotFoundError(f"Makefile not found at {dirs.makefile_path}")

    if not dirs.lbm_src_path.exists():
        raise FileNotFoundError(f"LBM src directory not found at {dirs.lbm_src_path}")

    # Update makefile HOME path to pixi environment
    makefile = Makefile(dirs.makefile_path)
    makefile.set_path("HOME", dirs.pixi_env_path)

    # Also update NCFDIR if NETCDF is enabled
    if enable_netcdf:
        makefile.set_path("NCFDIR", dirs.pixi_env_path)

    makefile.write()

    if verbose:
        logger.info("Updated makefile HOME to %s", dirs.pixi_env_path)

    # Set up environment variables
    env = os.environ.copy()
    env["HOME"] = str(dirs.pixi_env_path)
    if "PIXI_ENVIRONMENT" not in env:
        env["PIXI_ENVIRONMENT"] = str(dirs.pixi_env_path)

    # Change to LBM src directory and run make
    original_cwd = pathlib.Path.cwd()
    stdout = sys.stdout if verbose else subprocess.DEVNULL
    stderr = sys.stderr if verbose else subprocess.DEVNULL

    try:
        os.chdir(dirs.lbm_src_path)

        if verbose:
            logger.info("Changed to directory: %s", dirs.lbm_src_path)
            logger.info("Compiling LBM program...")

        # Build make command
        make_args = ["make", "-B", "GFORTRAN=1"]
        if enable_netcdf:
            make_args.append("NETCDF=1")

        result = subprocess.run(  # type: ignore[call-overload]
            make_args,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"LBM compilation failed with exit code {result.returncode}. "
                f"See output above for details."
            )

        if verbose:
            logger.info("LBM compilation completed successfully")

    finally:
        # Always return to original directory
        os.chdir(original_cwd)
