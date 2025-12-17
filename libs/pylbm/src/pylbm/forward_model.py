import os
import pathlib
import pdb
import subprocess
import sys
from typing import Optional

import xarray

from pyurbanair.base_forward_model import BaseForwardModel

from . import LBM_PATH
from .compile_program import compile_lbm, create_infile, identify_environment
from .stl_to_lbm import stl_to_lbm_geometry
from .write_to_fortran import (
    add_case_dimensions_to_mod_dimensions,
    add_case_to_select_statement,
    add_module_to_main,
)


class ForwardModel(BaseForwardModel):
    def __init__(
        self,
        rundir: pathlib.Path | None = None,
        stl_path: str | pathlib.Path | None = None,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        num_timesteps: int = 1000,
        bounds: (
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
        ) = None,
        results_dir: Optional[pathlib.Path] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(results_dir=results_dir)

        # Verbosity
        self.verbose = verbose
        self.stdout = None if self.verbose else subprocess.DEVNULL
        self.stderr = None if self.verbose else subprocess.DEVNULL

        # Default rundir is .temp/lbm relative to current working directory
        if rundir is None:
            rundir = pathlib.Path(".temp/lbm")

        # Convert to absolute path if relative
        if not rundir.is_absolute():
            rundir = pathlib.Path.cwd() / rundir

        self.rundir = rundir
        # Ensure rundir exists
        self.rundir.mkdir(parents=True, exist_ok=True)

        # Path to infile.in in rundir
        self.infile_path = self.rundir / "infile.in"

        # Case name
        self.case_name = "runcase"

        # Generate geometry file
        stl_to_lbm_geometry(
            stl_path=stl_path,  # type: ignore[arg-type]
            output_path=LBM_PATH / "src" / f"m_{self.case_name}.F90",  # type: ignore[operator]
            module_name=f"m_{self.case_name}",
            subroutine_name=self.case_name,
            nx=nx,  # type: ignore[arg-type]
            ny=ny,  # type: ignore[arg-type]
            nz=nz,  # type: ignore[arg-type]
            bounds=bounds,
        )

        # Add module to main.F90
        add_module_to_main(self.case_name)

        # Add case to select statement in main.F90
        add_case_to_select_statement(self.case_name)

        # Set case dimensions
        add_case_dimensions_to_mod_dimensions(self.case_name, nx, ny, nz)  # type: ignore[arg-type]

        # Compile program
        compile_lbm(rundir=self.rundir, case_name=self.case_name, verbose=self.verbose)

        # Create infile.in by running the executable
        create_infile(rundir=self.rundir)

        # Set number of timesteps
        self.num_timesteps = num_timesteps
        self._set_num_timesteps(self.num_timesteps)

    def _set_num_timesteps(self, num_timesteps: int) -> None:
        """Set the number of timesteps."""
        # Ensure infile.in exists
        if not self.infile_path.exists():
            raise FileNotFoundError(
                f"infile.in not found at {self.infile_path}. "
                f"Make sure create_infile() has been called."
            )
        # Read the infile
        with open(self.infile_path, "r") as f:
            lines = f.readlines()

        # Find and modify the line with nt1
        modified = False
        for i, line in enumerate(lines):
            if "! nt1" in line:
                # Replace the number at the beginning while preserving the comment
                # Format: "1000             ! nt1           : Final timestep"
                parts = line.split("!")
                if len(parts) >= 2:
                    comment = "!" + "!".join(parts[1:])  # Preserve the comment part
                    # Format the new line with proper spacing (match original format)
                    lines[i] = f"{num_timesteps}             {comment}"
                    modified = True
                    break

        if not modified:
            raise ValueError(
                f"Could not find 'nt1' line in infile.in at {self.infile_path}"
            )

        # Write the updated file
        with open(self.infile_path, "w") as f:
            f.writelines(lines)

    def run(self) -> None:
        """
        Run the LBM executable from the rundir.

        This executes the compiled boltzmann program from self.rundir,
        which will read infile.in and run the simulation.
        """
        # Find the pixi environment path to locate the executable
        _project_root = pathlib.Path(__file__).parent.parent.parent
        _repo_root = _project_root.parent.parent
        while _repo_root != _repo_root.parent:
            if (_repo_root / ".git").exists() or (_repo_root / ".gitmodules").exists():
                break
            _repo_root = _repo_root.parent

        # Identify the current pixi environment
        pixi_env_path = identify_environment(_repo_root, verbose=self.verbose)
        executable_path = pixi_env_path / "bin" / "boltzmann"

        if not executable_path.exists():
            raise FileNotFoundError(
                f"Executable not found at {executable_path}. "
                f"Make sure the program has been compiled successfully."
            )

        # Ensure infile.in exists
        if not self.infile_path.exists():
            raise FileNotFoundError(
                f"infile.in not found at {self.infile_path}. "
                f"Make sure create_infile() has been called."
            )

        if self.verbose:
            print(f"Executable: {executable_path}", file=sys.stderr)

        original_cwd = pathlib.Path.cwd()

        try:
            os.chdir(self.rundir)

            # Set up environment
            env = os.environ.copy()
            env["HOME"] = str(pixi_env_path)
            if "PIXI_ENVIRONMENT" not in env:
                env["PIXI_ENVIRONMENT"] = str(pixi_env_path)

            # Set stack size limit to unlimited
            # shell_cmd = f"ulimit -s unlimited && {executable_path}"
            shell_cmd = f"{executable_path}"
            result = subprocess.run(
                shell_cmd,
                shell=True,
                env=env,
                stderr=self.stderr,  # Capture stderr to see segfault details
                stdout=self.stdout,  # Also capture stdout to see any error messages
                text=True,
            )

        finally:
            # Always return to original directory
            os.chdir(original_cwd)

    def _set_inflow_settings(
        self,
        inflow_angle: float,
        velocity_magnitude: float,
    ) -> None:
        """Set the inflow settings."""
        # Ensure infile.in exists
        if not self.infile_path.exists():
            raise FileNotFoundError(
                f"infile.in not found at {self.infile_path}. "
                f"Make sure create_infile() has been called."
            )

        # Read the infile
        with open(self.infile_path, "r") as f:
            lines = f.readlines()

        # Find and modify the line with uini, udir
        modified = False
        for i, line in enumerate(lines):
            if "! uini, udir" in line:
                # Replace the first two floats with velocity_magnitude and inflow_angle
                # Format: "8.0 0.0          ! uini, udir    : Inflow wind velocity..."
                parts = line.split("!")
                if len(parts) >= 2:
                    comment = "!" + "!".join(parts[1:])  # Preserve the comment part
                    # Format the new line with proper spacing
                    lines[i] = (
                        f"{velocity_magnitude:.1f} {inflow_angle:.1f}          {comment}"
                    )
                    modified = True
                    break

        if not modified:
            raise ValueError(
                f"Could not find 'uini, udir' line in infile.in at {self.infile_path}"
            )

        # Write the updated file
        with open(self.infile_path, "w") as f:
            f.writelines(lines)

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset | None:
        """Run the LBM executable from the rundir."""

        if params is not None:
            if "inflow_angle" in params:
                self.inflow_angle = params.inflow_angle.item()
            if "velocity_magnitude" in params:
                self.velocity_magnitude = params.velocity_magnitude.item()

        self._set_inflow_settings(
            inflow_angle=self.inflow_angle,
            velocity_magnitude=self.velocity_magnitude,
        )

        self.run()

        sim_name = f"out{self.num_timesteps:06d}.nc"
        state = xarray.load_dataset(self.rundir / sim_name)

        # Add time dimension as the first dimension
        state = state.expand_dims("time", axis=0)
        return state
