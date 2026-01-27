import logging
import os
import pathlib
import pdb
import shutil
import subprocess
import time
from typing import Optional

import xarray

from pyurbanair.base_forward_model import BaseForwardModel

from . import UDALES_PATH
from .utils.clean_up_utils import clean_output_dir, clean_temp_dir
from .utils.config_utils import create_config_sh
from .utils.dir_utils import get_udales_directory_paths
from .utils.file_utils import copy_files
from .utils.namoptions_utils import rename_namoptions_file
from .utils.ncpu_utils import validate_and_sync_ncpu
from .utils.params_utils import apply_inflow_settings, merge_params
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
        matlab_bin: pathlib.Path = DEFAULT_MATLAB_BIN,
        save_only_last_timestep: bool = False,
        output_frequency: Optional[float] = None,
        params: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
        verbose: bool = True,
        temp_dir: Optional[pathlib.Path] = None,
        experiment_base_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Initialize the ForwardModel.

        Args:
            case_dir: The directory containing the original case files.
            experiment_name: The name of the experiment.
            ncpu: The number of CPUs to use.
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

        # Copy files from case_dir to experiment_dir
        copy_files(self.dirs.case_dir, self.dirs.experiment_dir)

        # Rename the namoptions file to have the experiment_name as its extension
        rename_namoptions_file(self.dirs.experiment_dir, self.dirs.experiment_name)

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
        apply_inflow_settings(self.params, self.dirs)

        if self.save_only_last_timestep:
            apply_save_only_last_timestep(self.dirs)
        elif self.output_frequency is not None:
            apply_output_frequency(self.dirs, self.output_frequency)

        logger.info(f"Experiment name: {self.dirs.experiment_name}")
        logger.info(f"Case dir: {self.dirs.case_dir}")
        logger.info(f"Temp dir: {self.dirs.temp_dir}")
        logger.info(f"Experiment base dir: {self.dirs.experiment_base_dir}")
        logger.info(f"Experiment dir: {self.dirs.experiment_dir}")
        logger.info(f"Output dir: {self.dirs.output_dir}")
        logger.info(f"NCPU: {self.ncpu}")
        logger.info(f"MATLAB bin: {self.matlab_bin}")

    def run_preprocessing(self, python_or_matlab: str = "python") -> None:
        """Run preprocessing."""

        logger.info("Running preprocessing...")

        clean_temp_dir(self.dirs)
        clean_output_dir(self.dirs)

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
    ) -> xarray.Dataset | None:
        """Run the forward model."""

        # Merge new params with existing params
        if params is not None:
            self.params = merge_params(self.params, params)

        apply_inflow_settings(self.params, self.dirs)

        logger.info("Running forward model...")
        command = [
            "bash",
            str(
                pathlib.Path(self.dirs.udales_root_path).joinpath(
                    "tools", "local_execute.sh"
                )
            ),
            str(self.dirs.experiment_dir),
        ]

        subprocess.run(command, check=True, stdout=self.stdout, stderr=self.stderr)

        output_file = self.dirs.output_dir.joinpath(
            self.dirs.experiment_name, f"fielddump.{self.dirs.experiment_name}.nc"
        )

        # Load into memory if save_in_memory is True
        if self.save_in_memory:
            state = xarray.open_dataset(
                output_file,
                engine="netcdf4",
            )
            state = state.load()
        else:
            outfile = self.dirs.results_dir / f"{sim_name}.nc"
            os.makedirs(str(self.dirs.results_dir), exist_ok=True)
            shutil.move(str(output_file), str(outfile))
            state = None

        if self.clean_output:
            clean_output_dir(self.dirs)

        return state
