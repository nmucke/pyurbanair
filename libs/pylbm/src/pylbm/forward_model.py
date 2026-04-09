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
from .utils.params_utils import (
    extract_initial_params,
    is_time_varying_params,
    remove_uvel_time_file,
    write_uvel_time_file,
)
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
        boundary_condition: str = "periodic",
        spinup_time: float = 0.0,
    ) -> None:
        super().__init__(results_dir=results_dir)

        self.spinup_time = spinup_time
        self._spinup_outputs = 0

        if boundary_condition not in ("periodic", "inflow_outflow"):
            raise ValueError(
                f"boundary_condition must be 'periodic' or 'inflow_outflow', "
                f"got '{boundary_condition}'"
            )
        self.boundary_condition = boundary_condition

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

        self.min_cell_size = min(dx, dy, dz)
        self.min_cell_size = np.round(self.min_cell_size, 1)

        # Set experiment dimensions in mod_dimensions.F90 (add or update experiment, set active)
        set_experiment(dirs=self.dirs, nx=nx, ny=ny, nz=nz)

        self.simulation_time = simulation_time
        self.output_frequency = output_frequency
        self.seconds_per_timestep: float | None = None
        # Derived during compile() once infile.in exists (C_t = C_l / C_u).
        self.num_timesteps = 0
        self.output_frequency_timesteps = 0
        # Warm-start override: when set, _set_scaling_factors uses this as nt0
        # instead of defaulting to 0. Consumed (reset to None) after each use.
        self._nt0_override: int | None = None

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

        # Set runtime controls in timestep units
        self._set_infile_value("experiment", self.dirs.experiment_name)
        self._set_infile_value("tecout", "3" if self.enable_netcdf else "0")

        # Apply x-direction boundary condition (y is always periodic: jbnd=0)
        ibnd = 0 if self.boundary_condition == "periodic" else 1
        self._set_infile_value("ibnd", ibnd)
        self._set_infile_value("jbnd", 0)

    def _set_scaling_factors(self, params: Optional[xarray.Dataset] = None) -> None:
        """Set the scaling factors for the LBM."""
        self._set_infile_value("C_l", self.min_cell_size)

        if params is not None:
            if is_time_varying_params(params):
                velocity_magnitude = float(params["velocity_magnitude"].max().item())
            else:
                velocity_magnitude = params["velocity_magnitude"].item()
            self.C_u = int(velocity_magnitude * 15)
        else:
            self.C_u = 75
        self._set_infile_value("C_u", self.C_u)

        self.seconds_per_timestep = self._compute_seconds_per_timestep()

        # Compute a fixed number of output steps independent of C_u, then derive
        # iout and num_timesteps so every ensemble member produces the same count.
        num_outputs = round(self.simulation_time / self.output_frequency)
        self.output_frequency_timesteps = max(
            1, round(self.output_frequency / self.seconds_per_timestep)
        )
        self.num_timesteps = self.output_frequency_timesteps * num_outputs

        if self.num_timesteps <= 0:
            raise ValueError("Resolved num_timesteps must be > 0.")

        # Extend run by spinup period (outputs produced during spinup are
        # discarded after collection in run_single).
        if self.spinup_time > 0:
            self._spinup_outputs = round(self.spinup_time / self.output_frequency)
        else:
            self._spinup_outputs = 0
        spinup_timesteps = self._spinup_outputs * self.output_frequency_timesteps
        total_timesteps = self.num_timesteps + spinup_timesteps

        if self._nt0_override is not None:
            nt0 = self._nt0_override
            self._nt0_override = None
        else:
            nt0 = 0
        self._set_infile_value("nt0", nt0)
        self._set_infile_value("nt1", nt0 + total_timesteps)
        self._set_infile_value("iout", self.output_frequency_timesteps)

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
            if nt0 < timestep <= nt1:
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
        """Apply the inflow settings to the forward model.

        For time-varying parameters (Dataset with a ``time`` dimension),
        writes ``uvel_time.dat`` for the Fortran code and sets initial
        static values in ``infile.in``.  For static parameters, removes
        any stale ``uvel_time.dat`` and applies the values directly.
        """
        if is_time_varying_params(params):
            write_uvel_time_file(
                params=params, dirs=self.dirs, spinup_time=self.spinup_time
            )
            initial_params = extract_initial_params(params)
            apply_inflow_settings(params=initial_params, dirs=self.dirs)
        else:
            remove_uvel_time_file(self.dirs)
            apply_inflow_settings(params=params, dirs=self.dirs)

    def save_results(self, state: xarray.Dataset, sim_name: str = "state") -> None:
        """Save simulation results to disk."""
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        """Remove netCDF output files from the output directory.

        This prevents stale files from being picked up by subsequent runs
        that may use a different output frequency (iout).
        """
        for output_file in self.dirs.output_dir.glob("out_*.nc"):
            output_file.unlink(missing_ok=True)

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
    ) -> xarray.Dataset:
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

        self._set_scaling_factors(params)

        # Remove stale output files before running to prevent files from a
        # previous run (which may have used a different iout) being collected.
        self._clean_output()

        self.run()

        output_files = self._get_output_files_for_current_run()
        state = [xarray.load_dataset(path, engine="netcdf4") for path in output_files]

        if len(state) > 1:
            state = xarray.concat(state, dim="time", join="override")
        else:
            state = state[0].expand_dims("time", axis=0)

        state = state.assign(x=self.x_grid, y=self.y_grid, z=self.z_grid)
        state = scale_velocity_to_physical(state, scale=self.C_u)

        if self._spinup_outputs > 0 and state.sizes["time"] > self._spinup_outputs:
            state = state.isel(time=slice(self._spinup_outputs, None))

        state = state.assign_coords(time=range(state.sizes["time"]))

        return state

    def disable_spinup(self) -> None:
        """Disable spinup so subsequent runs use only simulation_time."""
        self.spinup_time = 0.0
