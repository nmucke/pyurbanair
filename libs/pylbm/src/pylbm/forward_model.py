import logging
import os
import pathlib
import re
import subprocess
from typing import Optional, Union

logger = logging.getLogger(__name__)

import numpy as np
import xarray
from pylbm.utils import get_lbm_directory_paths

from pyurbanair.base_forward_model import BaseForwardModel

from .stl_to_lbm import stl_to_lbm_geometry
from .utils import Infile, apply_inflow_settings, compile_lbm, create_infile
from .utils.environment_utils import identify_environment
from .utils.infile_utils import _augment_runtime_library_paths
from .utils.mod_dimensions_utils import set_experiment
from .utils.state_utils import scale_velocity_to_physical


class ForwardModel(BaseForwardModel):
    def __init__(
        self,
        stl_path: str | pathlib.Path,
        rundir: pathlib.Path | None = None,
        nx: int = 120,
        ny: int = 120,
        nz: int = 8,
        simulation_time: float = 53.8,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (0, 160),
            (0, 160),
            (0, 40),
        ),
        output_frequency: float = 0.0538,
        results_dir: Optional[pathlib.Path] = None,
        verbose: bool = True,
        experiment_name: str = "runcase",
        cuda: bool = False,
        enable_netcdf: Optional[bool] = None,
    ) -> None:
        super().__init__(results_dir=results_dir)

        # Verbosity
        self.verbose = verbose
        self.cuda = cuda
        # Keep NETCDF enabled by default for both CPU and CUDA paths.
        self.enable_netcdf = True if enable_netcdf is None else enable_netcdf
        self.stdout = None if self.verbose else subprocess.DEVNULL
        self.stderr = None if self.verbose else subprocess.DEVNULL

        self.dirs = get_lbm_directory_paths(
            temp_dir=pathlib.Path(".temp"),
            case_dir=pathlib.Path("examples/lbm/experiments"),
            experiment_name=experiment_name,
        )

        # Generate geometry file
        stl_to_lbm_geometry(
            stl_path=stl_path,
            dirs=self.dirs,
            nx=nx,
            ny=ny,
            nz=nz,
            bounds=bounds,
        )

        # Compute cell size from bounds
        dx = (bounds[0][1] - bounds[0][0]) / nx
        dy = (bounds[1][1] - bounds[1][0]) / ny
        dz = (bounds[2][1] - bounds[2][0]) / nz
        self.x_grid = (np.arange(nx) + 0.5) * dx + bounds[0][0]
        self.y_grid = (np.arange(ny) + 0.5) * dy + bounds[1][0]
        self.z_grid = (np.arange(nz) + 0.5) * dz + bounds[2][0]

        # Set experiment dimensions in mod_dimensions.F90 (add or update experiment, set active)
        set_experiment(dirs=self.dirs, nx=nx, ny=ny, nz=nz)

        self.simulation_time = simulation_time
        self.output_frequency = output_frequency
        self.seconds_per_timestep: float | None = None
        # Derived during compile() once infile.in exists (C_t = C_l / C_u).
        self.num_timesteps = 0
        self.output_frequency_timesteps = 0

    def _compute_seconds_per_timestep(self) -> float:
        """Compute seconds per timestep from infile constants C_l/C_u."""
        infile = Infile(self.dirs.infile_path)
        c_l = infile.get_value_as_float("C_l")
        c_u = infile.get_value_as_float("C_u")
        if c_l is None or c_u is None:
            raise ValueError(
                "Could not read C_l/C_u from infile.in to compute timestep duration."
            )
        if c_u <= 0:
            raise ValueError("C_u in infile.in must be > 0.")
        return c_l / c_u

    def compile(self) -> None:
        """Compile the LBM program."""
        # Compile program
        compile_lbm(
            dirs=self.dirs,
            verbose=self.verbose,
            enable_cuda=self.cuda,
            enable_netcdf=self.enable_netcdf,
        )

        # Create infile.in by running the executable (only if it doesn't exist)
        if not self.dirs.infile_path.exists():
            create_infile(dirs=self.dirs, verbose=self.verbose)
        elif self.verbose:
            logger.info(
                "infile.in already exists at %s, skipping creation.",
                self.dirs.infile_path,
            )

        self.seconds_per_timestep = self._compute_seconds_per_timestep()
        self.num_timesteps = int(self.simulation_time / self.seconds_per_timestep)
        self.output_frequency_timesteps = int(
            self.output_frequency / self.seconds_per_timestep
        )
        if self.num_timesteps <= 0:
            raise ValueError("Resolved num_timesteps must be > 0.")
        if self.output_frequency_timesteps <= 0:
            raise ValueError("Resolved output frequency must be >= 1 timestep.")

        # Set runtime controls in timestep units
        self._set_infile_value("nt1", self.num_timesteps)
        self._set_infile_value("iout", self.output_frequency_timesteps)
        self._set_infile_value("experiment", self.dirs.experiment_name)
        self._set_infile_value("tecout", "3" if self.enable_netcdf else "0")

    def set_results_dir(self, results_dir: pathlib.Path | None) -> None:
        """Change results directory, updating both base and dirs dataclass."""
        super().set_results_dir(results_dir)
        self.dirs.results_dir = results_dir

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

    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        """Apply the inflow settings to the forward model."""
        apply_inflow_settings(params=params, dirs=self.dirs)

    def run(self) -> None:
        """
        Run the LBM executable from the rundir.

        This executes the compiled boltzmann program from self.rundir,
        which will read infile.in and run the simulation.
        """

        if self.verbose:
            logger.info("Executable: %s", self.dirs.executable_path)

        original_cwd = pathlib.Path.cwd()

        os.chdir(self.dirs.experiment_dir)

        # Set up environment
        env = os.environ.copy()
        env["HOME"] = str(self.dirs.pixi_env_path)
        if "PIXI_ENVIRONMENT" not in env:
            env["PIXI_ENVIRONMENT"] = str(self.dirs.pixi_env_path)
        _augment_runtime_library_paths(env=env, pixi_env_path=self.dirs.pixi_env_path)

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
        if not self.enable_netcdf:
            raise RuntimeError(
                "run_single requires NETCDF output, but this model was compiled with "
                "enable_netcdf=False (default for cuda=True). "
                "Either set enable_netcdf=True with an NVFORTRAN-compatible netcdf.mod, "
                "or call run() and process non-NETCDF diagnostics."
            )

        if params is not None:
            self._apply_inflow_settings(params)

        self.run()

        output_files = self._get_output_files_for_current_run()
        datasets = [
            xarray.load_dataset(path, engine="netcdf4") for path in output_files
        ]

        if len(datasets) > 1:
            state = xarray.concat(datasets, dim="time", join="override")
        else:
            state = datasets[0].expand_dims("time", axis=0)

        state = state.assign(x=self.x_grid, y=self.y_grid, z=self.z_grid)
        state = scale_velocity_to_physical(state)

        return state
