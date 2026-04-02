import logging
import os
import pathlib
import shutil
import subprocess
import time
from typing import Optional

import xarray

from pyurbanair.base_forward_model import BaseForwardModel

from . import LOCAL_EXECUTE_SCRIPT, UDALES_PATH
from .utils.clean_up_utils import clean_output_dir, clean_temp_dir
from .utils.config_utils import create_config_sh
from .utils.dir_utils import get_project_root, get_udales_directory_paths
from .utils.file_utils import copy_files
from .utils.namoptions_utils import NamoptionsFile, rename_namoptions_file
from .utils.ncpu_utils import validate_and_sync_ncpu
from .utils.params_utils import apply_inflow_settings, merge_params
from .utils.random_utils import apply_random_initial_condition
from .utils.save_frequency_utils import (
    apply_output_frequency,
    apply_save_only_last_timestep,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MATLAB_BIN = pathlib.Path("/Applications/MATLAB_R2025b.app/bin/matlab")

DEFAULT_TEMP_DIR = lambda cwd: pathlib.Path(f"{cwd}/.temp")

# Default parameter values as xarray.Dataset
DEFAULT_PARAMS = xarray.Dataset(
    data_vars={
        "inflow_angle": 45,
        "velocity_magnitude": 3,
        "pressure_gradient_magnitude": 0.0041912,
    },
)

DomainBounds = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


def _augment_runtime_library_paths(env: dict[str, str]) -> None:
    """Ensure runtime loader can find shared libraries in active pixi/conda env."""
    lib_paths: list[pathlib.Path] = []

    conda_prefix = env.get("CONDA_PREFIX")
    pixi_environment = env.get("PIXI_ENVIRONMENT")

    prefix_candidates: list[pathlib.Path] = []
    if conda_prefix:
        prefix_candidates.append(pathlib.Path(conda_prefix))
    if pixi_environment:
        pixi_path = pathlib.Path(pixi_environment)
        if pixi_path.exists():
            prefix_candidates.append(pixi_path)
        else:
            prefix_candidates.append(
                get_project_root() / ".pixi" / "envs" / pixi_environment
            )

    for prefix_path in prefix_candidates:
        if not prefix_path.exists():
            continue
        lib_dir = prefix_path / "lib"
        if lib_dir.exists():
            lib_paths.append(lib_dir)

        # Also include optional NVHPC-local netcdf-fortran install used by CUDA/LBM.
        local_netcdf_lib = prefix_path / ".nvhpc" / "netcdf-fortran" / "lib"
        if local_netcdf_lib.exists():
            lib_paths.append(local_netcdf_lib)

    if not lib_paths:
        return

    existing = env.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(str(p) for p in lib_paths)
    env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


class ForwardModel(BaseForwardModel):
    """
    Forward model class.

    The forward model is a wrapper around the uDALES code.
    """

    def __init__(
        self,
        case_dir: pathlib.Path,
        experiment_name: str = "300",
        ncpu: int = 4,
        simulation_time: float | None = None,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        bounds: DomainBounds | None = None,
        matlab_bin: pathlib.Path = DEFAULT_MATLAB_BIN,
        save_only_last_timestep: bool = False,
        output_frequency: Optional[float] = None,
        params: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
        verbose: bool = True,
        temp_dir: Optional[pathlib.Path] = None,
        experiment_base_dir: Optional[pathlib.Path] = None,
        random_initial_condition_args: Optional[dict] = None,
        boundary_condition: str = "periodic",
        spinup_time: float = 0.0,
    ) -> None:
        """
        Initialize the ForwardModel.

        Args:
            case_dir: The directory containing the original case files.
            experiment_name: The name of the experiment.
            ncpu: The number of CPUs to use.
            simulation_time: Total simulation runtime in seconds. If provided,
                writes &RUN runtime in namoptions.
            nx: Number of grid cells in x direction (maps to itot in namoptions).
            ny: Number of grid cells in y direction (maps to jtot in namoptions).
            nz: Number of grid cells in z direction (maps to ktot in namoptions).
            bounds: Domain bounds in the form
                ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
                Domain lengths are written to xlen/ylen/zsize in namoptions.
            matlab_bin: The path to the MATLAB binary.
            save_only_last_timestep: If True, only the last timestep will be saved. Overwrites save_frequency.
            output_frequency: The frequency at which the output will be saved.
            params: The parameters of the forward model.
                Currently, we only support the following parameters:
                - inflow_angle: The angle of the inflow wind speed in degrees (measured from positive x-axis).
                - velocity_magnitude: The magnitude of the inflow wind speed (m/s).
                - pressure_gradient_magnitude: The magnitude of the inflow pressure gradient (Pa/m).
            results_dir: The directory where the results will be saved.
            verbose: If True, print output from Fortran code execution. If False, suppress all output.
            temp_dir: The base temp directory (defaults to {cwd}/.temp).
            experiment_base_dir: The base directory for experiments (defaults to {temp_dir}/experiment).
        """
        super().__init__(results_dir=results_dir)

        # Verbose flag for controlling output
        self.verbose = verbose
        self.stdout = None if self.verbose else subprocess.DEVNULL
        self.stderr = None if self.verbose else subprocess.DEVNULL

        self.clean_output = True

        # Create directory paths dataclass with defaults or provided paths
        self.dirs = get_udales_directory_paths(
            case_dir=case_dir,
            experiment_name=experiment_name,
            udales_root_path=UDALES_PATH,  # type: ignore[arg-type]
            temp_dir=temp_dir,
            experiment_base_dir=experiment_base_dir,
            results_dir=results_dir,
        )

        # Save only the last timestep
        self.save_only_last_timestep = save_only_last_timestep

        # Save frequency
        if save_only_last_timestep:
            self.output_frequency = None
        else:
            self.output_frequency = output_frequency

        # MATLAB binary
        self.matlab_bin = matlab_bin

        # Initialize params by merging provided params with defaults
        self.params = merge_params(
            existing_params=DEFAULT_PARAMS,
            new_params=params,
        )
        if self.params is None:
            raise ValueError("ForwardModel requires at least one inflow parameter.")

        # Copy files from case_dir to experiment_dir
        copy_files(self.dirs.case_dir, self.dirs.experiment_dir)

        # Rename the namoptions file to have the experiment_name as its extension
        rename_namoptions_file(self.dirs.experiment_dir, self.dirs.experiment_name)

        self.bounds: DomainBounds | None = None

        self.spinup_time = spinup_time
        self._simulation_time = simulation_time

        self._apply_runtime_override(simulation_time=simulation_time)
        self._apply_domain_overrides(nx=nx, ny=ny, nz=nz, bounds=bounds)

        # Apply boundary conditions (x-direction configurable, y always periodic)
        if boundary_condition not in ("periodic", "inflow_outflow"):
            raise ValueError(
                f"boundary_condition must be 'periodic' or 'inflow_outflow', "
                f"got '{boundary_condition}'"
            )
        self.boundary_condition = boundary_condition
        self._apply_boundary_condition()

        # Validate and sync NCPU with nprocx * nprocy from namoptions
        self.ncpu = validate_and_sync_ncpu(
            dirs=self.dirs,
            ncpu=ncpu,
        )

        # Create a config.sh file where the environment variables are set
        create_config_sh(
            dirs=self.dirs,
            matlab_bin=self.matlab_bin,
            ncpu=self.ncpu,
        )

        # Apply inflow settings
        if self.params is None:
            raise ValueError("ForwardModel parameters are unexpectedly unset.")
        apply_inflow_settings(self.params, self.dirs)

        if self.save_only_last_timestep:
            apply_save_only_last_timestep(self.dirs)
        elif self.output_frequency is not None:
            apply_output_frequency(self.dirs, self.output_frequency)

        if random_initial_condition_args is not None:
            apply_random_initial_condition(self.dirs, random_initial_condition_args)

        logger.info(f"Experiment name: {self.dirs.experiment_name}")
        logger.info(f"Case dir: {self.dirs.case_dir}")
        logger.info(f"Temp dir: {self.dirs.temp_dir}")
        logger.info(f"Experiment base dir: {self.dirs.experiment_base_dir}")
        logger.info(f"Experiment dir: {self.dirs.experiment_dir}")
        logger.info(f"Output dir: {self.dirs.output_dir}")
        logger.info(f"NCPU: {self.ncpu}")
        logger.info(f"MATLAB bin: {self.matlab_bin}")

    def _apply_runtime_override(self, simulation_time: float | None) -> None:
        """Apply optional simulation runtime override to namoptions.

        When spinup_time > 0 the effective runtime written to namoptions is
        ``simulation_time + spinup_time``; the extra output produced during
        the spinup window is trimmed in ``run_single``.
        """
        if simulation_time is None:
            return
        if simulation_time <= 0:
            raise ValueError("simulation_time must be > 0.")

        effective_time = simulation_time + self.spinup_time

        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)
        namoptions.set_value("RUN", "runtime", effective_time)
        namoptions.write()

    def _apply_domain_overrides(
        self,
        nx: int | None,
        ny: int | None,
        nz: int | None,
        bounds: DomainBounds | None,
    ) -> None:
        """Apply optional domain overrides to namoptions."""
        provided_any = any(v is not None for v in (nx, ny, nz, bounds))
        if not provided_any:
            return

        if nx is None or ny is None or nz is None or bounds is None:
            raise ValueError(
                "If one of nx/ny/nz/bounds is provided, all four must be provided."
            )

        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("nx, ny, and nz must all be positive integers.")

        for axis_name, axis_bounds in zip(("x", "y", "z"), bounds):
            if axis_bounds[1] <= axis_bounds[0]:
                raise ValueError(
                    f"Invalid {axis_name} bounds: upper bound must be greater than lower bound."
                )

        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)
        namoptions.set_value("DOMAIN", "itot", nx)
        namoptions.set_value("DOMAIN", "jtot", ny)
        namoptions.set_value("DOMAIN", "ktot", nz)
        namoptions.set_value("DOMAIN", "xlen", bounds[0][1] - bounds[0][0])
        namoptions.set_value("DOMAIN", "ylen", bounds[1][1] - bounds[1][0])
        namoptions.set_value("INPS", "zsize", bounds[2][1] - bounds[2][0])
        namoptions.write()

        self.bounds = bounds

        # When the domain has non-zero lower bounds, the STL geometry must be
        # shifted so that it is correctly positioned within the [0, length]
        # computational domain.  uDALES always starts its grid at 0, so an
        # STL vertex originally at x=0 needs to move to x=-xmin (e.g. +100
        # when xmin=-100) inside the enlarged domain.
        x_offset = -bounds[0][0]
        y_offset = -bounds[1][0]
        z_offset = -bounds[2][0]
        if x_offset != 0.0 or y_offset != 0.0 or z_offset != 0.0:
            self._shift_stl_geometry(
                namoptions_path, (x_offset, y_offset, z_offset)
            )

    def _shift_stl_geometry(
        self,
        namoptions_path: pathlib.Path,
        offsets: tuple[float, float, float],
    ) -> None:
        """Shift STL geometry vertices so buildings sit correctly in the enlarged domain.

        Args:
            namoptions_path: Path to the namoptions file (used to read the STL filename).
            offsets: (x_offset, y_offset, z_offset) to add to every vertex.
        """
        import trimesh

        namoptions = NamoptionsFile(namoptions_path)
        stl_filename = namoptions.get_value("INPS", "stl_file")
        if stl_filename is None:
            logger.warning("No stl_file found in namoptions; skipping STL shift.")
            return

        stl_path = self.dirs.experiment_dir / stl_filename.strip().strip("'\"")
        if not stl_path.exists():
            logger.warning("STL file %s not found; skipping STL shift.", stl_path)
            return

        mesh = trimesh.load(stl_path)
        mesh.vertices[:, 0] += offsets[0]
        mesh.vertices[:, 1] += offsets[1]
        mesh.vertices[:, 2] += offsets[2]
        mesh.export(stl_path)

        logger.info(
            "Shifted STL vertices by (%.2f, %.2f, %.2f) for negative-bound domain.",
            *offsets,
        )

    def _apply_boundary_condition(self) -> None:
        """Apply x-direction boundary condition to namoptions (y is always periodic)."""
        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)
        bcxm = 1 if self.boundary_condition == "periodic" else 2
        namoptions.set_value("BC", "BCxm", bcxm)
        namoptions.set_value("BC", "BCym", 1)
        namoptions.write()

    def set_results_dir(self, results_dir: pathlib.Path | None) -> None:
        """Change results directory, updating both base and dirs dataclass."""
        super().set_results_dir(results_dir)
        self.dirs.results_dir = results_dir

    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        """Apply the inflow settings to the forward model."""
        if params is not None:
            self.params = merge_params(self.params, params)

        if self.params is None:
            raise ValueError("ForwardModel parameters are unexpectedly unset.")

        apply_inflow_settings(params=self.params, dirs=self.dirs)

    def save_results(self, state: xarray.Dataset, sim_name: str = "state") -> None:
        """Save simulation results to disk."""
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        """Clean the output directory."""
        clean_output_dir(self.dirs)

    def run_preprocessing(self, python_or_matlab: str = "python") -> None:
        """Run preprocessing."""

        logger.info("Running preprocessing...")

        clean_temp_dir(self.dirs)
        self._clean_output()

        if python_or_matlab == "python":
            # Use Python-based preprocessing script
            script_path = (
                pathlib.Path(__file__).parent.parent.parent
                / "shell_scripts"
                / "write_inputs.sh"
            )

            command = [
                "bash",
                str(script_path),
                str(self.dirs.experiment_dir),
            ]
            env = os.environ.copy()
            # Set environment variables needed by the script
            env["DA_EXPDIR"] = str(self.dirs.experiment_base_dir)
            env["DA_TOOLSDIR"] = str(
                pathlib.Path(self.dirs.udales_root_path).joinpath("tools")
            )
            _augment_runtime_library_paths(env)

        elif python_or_matlab == "matlab":
            # Use MATLAB-based preprocessing script
            command = [
                "bash",
                str(
                    pathlib.Path(self.dirs.udales_root_path).joinpath(
                        "tools", "write_inputs.sh"
                    )
                ),
                str(self.dirs.experiment_dir),
            ]
            # Add MATLAB bin directory to PATH so the script can find 'matlab'
            env = os.environ.copy()
            matlab_bin_dir = str(pathlib.Path(self.matlab_bin).parent)
            env["PATH"] = f"{matlab_bin_dir}:{env.get('PATH', '')}"
            _augment_runtime_library_paths(env)

        subprocess.run(
            command, check=True, env=env, stdout=self.stdout, stderr=self.stderr
        )

        # Wait for MATLAB preprocessing to complete if using MATLAB
        if python_or_matlab == "matlab":
            time.sleep(90)

        logger.info("Preprocessing completed.")

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        """Run the forward model."""

        self._apply_inflow_settings(params=params)

        logger.info("Running forward model...")
        command = [
            "bash",
            str(LOCAL_EXECUTE_SCRIPT),
            str(self.dirs.experiment_dir),
        ]

        env = os.environ.copy()
        _augment_runtime_library_paths(env)
        subprocess.run(
            command,
            check=True,
            env=env,
            stdout=self.stdout,
            stderr=self.stderr,
        )

        # Check for merged file first (multi-processor case after gather_outputs.sh)
        output_file = self.dirs.output_dir.joinpath(
            self.dirs.experiment_name, f"fielddump.{self.dirs.experiment_name}.nc"
        )

        # If merged file doesn't exist, check for single-processor file
        # (gather_outputs.sh doesn't merge when there's only one processor)
        if not output_file.exists():
            single_proc_file = self.dirs.output_dir.joinpath(
                self.dirs.experiment_name,
                f"fielddump.000.000.{self.dirs.experiment_name}.nc",
            )
            if single_proc_file.exists():
                output_file = single_proc_file

        # Load into memory if save_in_memory is True
        state = xarray.open_dataset(
            output_file,
            engine="netcdf4",
        )

        # Shift coordinates by lower-bound offset so that the state reflects
        # physical domain coordinates (matching pylbm behaviour for negative bounds).
        if self.bounds is not None:
            x_offset = self.bounds[0][0]
            y_offset = self.bounds[1][0]
            z_offset = self.bounds[2][0]
            coord_updates = {}
            for coord_name, offset in [
                ("xt", x_offset), ("xm", x_offset),
                ("yt", y_offset), ("ym", y_offset),
                ("zt", z_offset), ("zm", z_offset),
            ]:
                if coord_name in state.coords:
                    coord_updates[coord_name] = state.coords[coord_name].values + offset
            if coord_updates:
                state = state.assign_coords(coord_updates)

        if self.spinup_time > 0 and self.output_frequency is not None:
            spinup_outputs = round(self.spinup_time / self.output_frequency)
            state = state.isel(time=slice(spinup_outputs, None))
            if "time" in state.coords:
                state = state.assign_coords(time=state.time - state.time.values[0])

        return state

    def disable_spinup(self) -> None:
        """Disable spinup so subsequent runs use only simulation_time."""
        self.spinup_time = 0.0
        if self._simulation_time is not None:
            namoptions_path = (
                self.dirs.experiment_dir
                / f"namoptions.{self.dirs.experiment_name}"
            )
            namoptions = NamoptionsFile(namoptions_path)
            namoptions.set_value("RUN", "runtime", self._simulation_time)
            namoptions.write()
