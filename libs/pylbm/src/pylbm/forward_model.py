import os
import pathlib
import pdb
import subprocess
import sys
from typing import Optional, Union

import xarray
from pylbm.utils import get_lbm_directory_paths

from pyurbanair.base_forward_model import BaseForwardModel

from .stl_to_lbm import stl_to_lbm_geometry
from .utils import Infile, apply_inflow_settings, compile_lbm, create_infile
from .utils.environment_utils import identify_environment
from .utils.mod_dimensions_utils import set_experiment


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
        experiment_name: str = "runcase",
    ) -> None:
        super().__init__(results_dir=results_dir)

        # Verbosity
        self.verbose = verbose
        self.stdout = None if self.verbose else subprocess.DEVNULL
        self.stderr = None if self.verbose else subprocess.DEVNULL

        self.dirs = get_lbm_directory_paths(
            temp_dir=pathlib.Path(".temp"),
            case_dir=pathlib.Path("examples/lbm/experiments"),
            experiment_name=experiment_name,
        )

        # Generate geometry file
        stl_to_lbm_geometry(
            stl_path=stl_path,  # type: ignore[arg-type]
            dirs=self.dirs,
            nx=nx,  # type: ignore[arg-type]
            ny=ny,  # type: ignore[arg-type]
            nz=nz,  # type: ignore[arg-type]
            bounds=bounds,
        )

        # Set experiment dimensions in mod_dimensions.F90 (add or update experiment, set active)
        if nx is not None and ny is not None and nz is not None:
            set_experiment(dirs=self.dirs, nx=nx, ny=ny, nz=nz)

        # Compile program
        compile_lbm(dirs=self.dirs, verbose=self.verbose)

        # Create infile.in by running the executable (only if it doesn't exist)
        if not self.dirs.infile_path.exists():
            create_infile(dirs=self.dirs, verbose=self.verbose)
        elif self.verbose:
            print(
                f"infile.in already exists at {self.dirs.infile_path}, skipping creation.",
                file=sys.stderr,
            )

        # Set number of timesteps
        self.num_timesteps = num_timesteps
        self._set_infile_value("nt1", self.num_timesteps)
        self._set_infile_value("experiment", self.dirs.experiment_name)
        self._set_infile_value("tecout", "3")

    def _set_infile_value(self, key: str, value: Union[str, int, float, bool]) -> None:
        """
        Set a value in infile.in by key.

        This is a reusable helper method for updating any value in the infile.in file.

        Args:
            key: The key name in infile.in (e.g. "nt1", "experiment", "tecout").
            value: The value to set (will be converted to string if needed).
        """
        infile = Infile(self.dirs.infile_path)
        infile.set_value(key, value)
        infile.write()

    def run(self) -> None:
        """
        Run the LBM executable from the rundir.

        This executes the compiled boltzmann program from self.rundir,
        which will read infile.in and run the simulation.
        """

        if self.verbose:
            print(f"Executable: {self.dirs.executable_path}", file=sys.stderr)

        original_cwd = pathlib.Path.cwd()

        os.chdir(self.dirs.experiment_dir)

        # Set up environment
        env = os.environ.copy()
        env["HOME"] = str(self.dirs.pixi_env_path)
        if "PIXI_ENVIRONMENT" not in env:
            env["PIXI_ENVIRONMENT"] = str(self.dirs.pixi_env_path)

        # Set stack size limit to unlimited
        # shell_cmd = f"ulimit -s unlimited && {executable_path}"
        shell_cmd = f"{self.dirs.executable_path}"
        _ = subprocess.run(
            shell_cmd,
            shell=True,
            env=env,
            stderr=self.stderr,
            stdout=self.stdout,
            text=True,
        )

        # Always return to original directory
        os.chdir(original_cwd)

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the LBM executable from the rundir."""

        if params is not None:
            apply_inflow_settings(params=params, dirs=self.dirs)

        self.run()

        # sim_name = f"out_0000_{self.num_timesteps:06d}.nc"
        sim_name = f"out_0000_F000000.nc"

        state = xarray.load_dataset(self.dirs.output_dir / sim_name, engine="netcdf4")

        # Add time dimension as the first dimension
        state = state.expand_dims("time", axis=0)
        return state
