import os
import pathlib
import pdb
import re
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
        output_frequency: float = 1.0,
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
        self.output_frequency = output_frequency
        self._set_infile_value("nt1", self.num_timesteps)
        self._set_infile_value("iout", int(self.output_frequency))
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

    def _get_infile_int_value(self, key: str, default: int) -> int:
        """Read an integer value from infile.in, with fallback to default."""
        infile = Infile(self.dirs.infile_path)
        value = infile.get_value_as_int(key)
        return value if value is not None else default

    def _get_output_files_for_current_run(self) -> list[pathlib.Path]:
        """
        Return output netCDF files corresponding to the configured timestep range.

        This supports both cold-start runs (nt0=0) and warm-start runs (nt0>0).
        """
        nt0 = self._get_infile_int_value("nt0", 0)
        nt1 = self._get_infile_int_value("nt1", self.num_timesteps)

        output_files: list[tuple[int, pathlib.Path]] = []
        for path in self.dirs.output_dir.glob("out_0000_F*.nc"):
            match = re.search(r"_F(\d+)$", path.stem)
            if match is None:
                continue
            timestep = int(match.group(1))
            if nt0 <= timestep <= nt1:
                output_files.append((timestep, path))

        output_files = sorted(output_files, key=lambda x: x[0])
        if output_files:
            return [path for _, path in output_files]

        # Fallback to expected final file
        expected_file = self.dirs.output_dir / f"out_0000_F{nt1:06d}.nc"
        if expected_file.exists():
            return [expected_file]

        raise FileNotFoundError(
            f"No LBM output files found in {self.dirs.output_dir} for timestep range "
            f"[{nt0}, {nt1}]"
        )

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

        output_files = self._get_output_files_for_current_run()
        datasets = [
            xarray.load_dataset(path, engine="netcdf4") for path in output_files
        ]

        if len(datasets) > 1:
            state = xarray.concat(datasets, dim="time", join="override")
        else:
            state = datasets[0].expand_dims("time", axis=0)

        if self.save_on_disk and self.results_dir is not None:
            outfile = self.results_dir / f"{sim_name}.nc"
            state.to_netcdf(str(outfile))
            return None

        return state
