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
from .inflow_utils import angle_to_pressure_gradient, angle_to_velocity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MATLAB_BIN = pathlib.Path(
    "/Applications/MATLAB_R2025b.app/bin/matlab"
)

DEFAULT_TEMP_DIR = lambda cwd: pathlib.Path(f"{cwd}/.temp/experiments")


def create_dir(
    dir_path: pathlib.Path,
) -> pathlib.Path:
    """Create a temporary directory in the given directory."""
    os.makedirs(pathlib.Path(dir_path), exist_ok=True)
    return pathlib.Path(dir_path)


def move_files_to_temp_dir(
    experiment_dir: pathlib.Path, temp_dir: pathlib.Path
) -> None:
    """Move files from experiment_dir to temp_dir."""
    experiment_path = pathlib.Path(experiment_dir)
    temp_dir_path = temp_dir
    for item in experiment_path.iterdir():
        target = temp_dir_path / item.name
        if item.is_file():
            target.write_bytes(item.read_bytes())
        elif item.is_dir():
            # Remove target if it exists, then copy
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)


class ForwardModel(BaseForwardModel):
    """
    Forward model class.

    The forward model is a wrapper around the uDALES code.
    """

    def __init__(
        self,
        experiment_dir: pathlib.Path,
        ncpu: int = 4,
        matlab_bin: pathlib.Path = DEFAULT_MATLAB_BIN,
        save_only_last_timestep: bool = False,
        params: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
        verbose: bool = True,
        temp_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Initialize the ForwardModel.

        Args:
            experiment_dir: The directory containing the experiment.
            work_dir: The directory where the output will be saved.
            ncpu: The number of CPUs to use.
            matlab_bin: The path to the MATLAB binary.
            save_only_last_timestep: If True, only the last timestep will be saved.
            params: The parameters of the forward model.
                Currently, we only support the following parameters:
                - inflow_angle: The angle of the inflow wind speed in degrees (measured from positive x-axis).
                - velocity_magnitude: The magnitude of the inflow wind speed (m/s).
                - pressure_gradient_magnitude: The magnitude of the inflow pressure gradient (Pa/m).
            results_dir: The directory where the results will be saved.
            verbose: If True, print output from Fortran code execution. If False, suppress all output.
        """
        super().__init__(results_dir=results_dir)

        # UDALES root path where the udales code is located
        self.udales_root_path = UDALES_PATH

        # Current working directory
        self.cwd = pathlib.Path(__file__).parent.parent.parent.parent.parent

        # Save only the last timestep
        self.save_only_last_timestep = save_only_last_timestep

        # Experiment name
        self.experiment_name = str(experiment_dir)[-3:]

        # Experiment directory where the experiment is stored
        self.experiment_dir = experiment_dir

        # Temporary directory where the experiment is stored
        if temp_dir is None:
            self.temp_dir: pathlib.Path = create_dir(
                pathlib.Path(f"{self.cwd}/.temp/experiments/{self.experiment_name}")
            )
        else:
            self.temp_dir = temp_dir

        # Output directory where the intermediate udales outputs will be saved
        self.work_dir: pathlib.Path = create_dir(
            pathlib.Path(f"{self.cwd}/.temp/outputs")
        )

        # self.work_dir = work_dir
        self.ncpu = ncpu

        # MATLAB binary
        self.matlab_bin = matlab_bin

        # Parameters
        self.params = params

        # Verbose flag for controlling output
        self.verbose = verbose
        self.stdout = None if self.verbose else subprocess.DEVNULL
        self.stderr = None if self.verbose else subprocess.DEVNULL

        logger.info(f"Experiment name: {self.experiment_name}")
        logger.info(f"Temp dir: {self.temp_dir}")
        logger.info(f"Experiment dir: {self.experiment_dir}")
        logger.info(f"Work dir: {self.work_dir}")
        logger.info(f"NCPU: {self.ncpu}")
        logger.info(f"MATLAB bin: {self.matlab_bin}")

        # Move files from experiment_dir to temp_dir
        move_files_to_temp_dir(self.experiment_dir, self.temp_dir)

        # Validate and sync NCPU with nprocx * nprocy from namoptions
        self._validate_and_sync_ncpu()

        # Create a config.sh file where the environment variables are set
        self._create_config_sh()

        # Apply inflow settings if provided
        # Only apply if we have at least an angle and one magnitude
        if self.params is not None:
            self._apply_inflow_settings(
                inflow_angle=self.params.inflow_angle.item(),
                velocity_magnitude=self.params.velocity_magnitude.item(),
                pressure_gradient_magnitude=self.params.pressure_gradient_magnitude.item(),
            )

        if self.save_only_last_timestep:
            self._apply_save_only_last_timestep()

    def _update_prof_file(self, u0: float | None, v0: float | None) -> None:
        """Update prof.inp.* file with new u0 and v0 values."""
        prof_path = pathlib.Path(self.temp_dir) / f"prof.inp.{self.experiment_name}"

        if not prof_path.exists():
            logger.warning(
                f"prof.inp.{self.experiment_name} not found, skipping update"
            )
            return

        if u0 is None and v0 is None:
            return

        # Read the file
        lines = []
        with open(prof_path, "r") as f:
            lines = f.readlines()

        # Update data lines (skip header lines starting with #)
        output_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                # Keep header and empty lines as is
                output_lines.append(line)
            else:
                # Parse the data line: z thl qt u v tke
                parts = stripped.split()
                if len(parts) >= 6:
                    z = float(parts[0])
                    thl = float(parts[1])
                    qt = float(parts[2])
                    u = u0 if u0 is not None else float(parts[3])
                    v = v0 if v0 is not None else float(parts[4])
                    tke = float(parts[5])
                    # Format to match MATLAB output: %-20.15f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f
                    output_lines.append(
                        f"{z:20.15f} {thl:12.6f} {qt:12.6f} {u:12.6f} {v:12.6f} {tke:12.6f}\n"
                    )
                else:
                    # Keep line as is if format is unexpected
                    output_lines.append(line)

        # Write the updated file
        with open(prof_path, "w") as f:
            f.writelines(output_lines)

        logger.info(f"Updated prof.inp.{self.experiment_name} with u0={u0}, v0={v0}")

    def _update_lscale_file(
        self,
        u0: Optional[float] = None,
        v0: Optional[float] = None,
        dpdx: Optional[float] = None,
        dpdy: Optional[float] = None,
    ) -> None:
        """Update lscale.inp.* file with new u0, v0, dpdx, and dpdy values."""
        lscale_path = pathlib.Path(self.temp_dir) / f"lscale.inp.{self.experiment_name}"

        if not lscale_path.exists():
            logger.warning(
                f"lscale.inp.{self.experiment_name} not found, skipping update"
            )
            return

        if u0 is None and v0 is None and dpdx is None and dpdy is None:
            return

        # Read the file
        lines = []
        with open(lscale_path, "r") as f:
            lines = f.readlines()

        # Update data lines (skip header lines starting with #)
        output_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                # Keep header and empty lines as is
                output_lines.append(line)
            else:
                # Parse the data line: z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad
                parts = stripped.split()
                if len(parts) >= 10:
                    z = float(parts[0])
                    uq = u0 if u0 is not None else float(parts[1])
                    vq = v0 if v0 is not None else float(parts[2])
                    pqx = dpdx if dpdx is not None else float(parts[3])
                    pqy = dpdy if dpdy is not None else float(parts[4])
                    wfls = float(parts[5])
                    dqtdxls = float(parts[6])
                    dqtdyls = float(parts[7])
                    dqtdtls = float(parts[8])
                    dthlrad = float(parts[9])
                    # Format to match MATLAB output: %-20.15f %-12.6f %-12.6f %-12.9f %-12.6f %-15.9f %-12.6f %-12.6f %-12.6f %-17.12f
                    output_lines.append(
                        f"{z:20.15f} {uq:12.6f} {vq:12.6f} {pqx:12.9f} {pqy:12.6f} "
                        f"{wfls:15.9f} {dqtdxls:12.6f} {dqtdyls:12.6f} {dqtdtls:12.6f} {dthlrad:17.12f}\n"
                    )
                else:
                    # Keep line as is if format is unexpected
                    output_lines.append(line)

        # Write the updated file
        with open(lscale_path, "w") as f:
            f.writelines(output_lines)

        logger.info(
            f"Updated lscale.inp.{self.experiment_name} with u0={u0}, v0={v0}, dpdx={dpdx}, dpdy={dpdy}"
        )

    def _apply_inflow_settings(
        self,
        inflow_angle: float,
        velocity_magnitude: float,
        pressure_gradient_magnitude: float,
    ) -> None:
        """Apply the inflow settings to namoptions file and update affected input files."""
        namoptions_path = (
            pathlib.Path(self.temp_dir) / f"namoptions.{self.experiment_name}"
        )

        self.inflow_angle = inflow_angle
        self.velocity_magnitude = velocity_magnitude
        self.pressure_gradient_magnitude = pressure_gradient_magnitude

        self.params = xarray.Dataset(
            data_vars={
                "inflow_angle": inflow_angle,
                "velocity_magnitude": velocity_magnitude,
                "pressure_gradient_magnitude": pressure_gradient_magnitude,
            },
        )

        # Calculate velocity and pressure gradient components from angle and magnitudes
        u0, v0 = angle_to_velocity(self.inflow_angle, self.velocity_magnitude)
        dpdx, dpdy = angle_to_pressure_gradient(
            self.inflow_angle, self.pressure_gradient_magnitude
        )

        # Read the file
        lines = []
        with open(namoptions_path, "r") as f:
            lines = f.readlines()

        # Process the file to update the &INPS section
        output_lines = []
        in_inps = False
        inps_section_found = False
        u0_updated = False
        v0_updated = False
        dpdx_updated = False
        dpdy_updated = False

        for line in lines:
            stripped = line.strip()

            # Check if we're entering or leaving the &INPS section
            if stripped.startswith("&INPS"):
                in_inps = True
                inps_section_found = True
                output_lines.append(line)
            elif stripped.startswith("/") and in_inps:
                # We're leaving the &INPS section, add any missing values before the closing /
                if self.velocity_magnitude is not None and not u0_updated:
                    output_lines.append(f"u0           = {u0:.7f}\n")
                if self.velocity_magnitude is not None and not v0_updated:
                    output_lines.append(f"v0           = {v0:.7f}\n")
                if self.pressure_gradient_magnitude is not None and not dpdx_updated:
                    output_lines.append(f"dpdx         = {dpdx:.7f}\n")
                if self.pressure_gradient_magnitude is not None and not dpdy_updated:
                    output_lines.append(f"dpdy         = {dpdy:.7f}\n")
                output_lines.append(line)
                in_inps = False
            elif in_inps:
                # We're in the &INPS section, check if this line needs to be updated
                if u0 is not None and "u0" in stripped and "=" in stripped:
                    output_lines.append(f"u0           = {u0:.7f}\n")
                    u0_updated = True
                elif v0 is not None and "v0" in stripped and "=" in stripped:
                    output_lines.append(f"v0           = {v0:.7f}\n")
                    v0_updated = True
                elif dpdx is not None and "dpdx" in stripped and "=" in stripped:
                    output_lines.append(f"dpdx         = {dpdx:.7f}\n")
                    dpdx_updated = True
                elif dpdy is not None and "dpdy" in stripped and "=" in stripped:
                    output_lines.append(f"dpdy         = {dpdy:.7f}\n")
                    dpdy_updated = True
                else:
                    # Keep the original line
                    output_lines.append(line)
            else:
                # Not in &INPS section, keep the line as is
                output_lines.append(line)

        # If &INPS section was not found, add it at the end
        if not inps_section_found:
            output_lines.append("\n&INPS\n")
            output_lines.append(f"u0           = {u0:.7f}\n")
            output_lines.append(f"v0           = {v0:.7f}\n")
            output_lines.append(f"dpdx         = {dpdx:.7f}\n")
            output_lines.append(f"dpdy         = {dpdy:.7f}\n")
            output_lines.append("/\n")

        # Write the updated file
        with open(namoptions_path, "w") as f:
            f.writelines(output_lines)

        # Update the affected input files
        self._update_prof_file(u0, v0)
        self._update_lscale_file(u0, v0, dpdx, dpdy)

        logger.info(
            f"Updated inflow settings: angle={self.inflow_angle}°, "
            f"u0={u0:.7f}, v0={v0:.7f}, dpdx={dpdx:.7f}, dpdy={dpdy:.7f}"
        )

    def _apply_save_only_last_timestep(self) -> None:
        """Apply the save_only_last_timestep flag."""
        # Set tfielddump = runtime in namoptions.{self.experiment_name}
        namoptions_path = (
            pathlib.Path(self.temp_dir) / f"namoptions.{self.experiment_name}"
        )
        lines = []
        runtime_value = None
        with open(namoptions_path, "r") as f:
            # First, parse out the runtime value from the &RUN section
            in_run = False
            for line in f:
                stripped = line.strip()
                if stripped.startswith("&RUN"):
                    in_run = True
                elif stripped.startswith("/"):
                    if in_run:
                        in_run = False
                elif in_run and "runtime" in stripped and "=" in stripped:
                    # example: runtime      = 5.
                    try:
                        right = stripped.split("=")[1]
                        runtime_value = right.strip().rstrip(".")
                        if "." in right.strip():
                            runtime_value = right.strip()
                    except Exception:
                        pass
                lines.append(line)
        # If we found the runtime_value, rewrite the file with tfielddump set to runtime under &OUTPUT
        if runtime_value is not None:
            output_lines = []
            in_output = False
            tfielddump_replaced = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("&OUTPUT"):
                    in_output = True
                elif stripped.startswith("/") and in_output:
                    if not tfielddump_replaced:
                        output_lines.append(f"tfielddump   = {runtime_value}\n")
                        tfielddump_replaced = True
                    in_output = False
                if in_output and "tfielddump" in stripped and "=" in stripped:
                    if not tfielddump_replaced:
                        output_lines.append(f"tfielddump   = {runtime_value}\n")
                        tfielddump_replaced = True
                else:
                    output_lines.append(line)
            with open(namoptions_path, "w") as f:
                f.writelines(output_lines)

    def _validate_and_sync_ncpu(self) -> None:
        """
        Validate and synchronize NCPU with nprocx * nprocy from namoptions.
        
        uDALES requires: nprocx * nprocy = NCPU
        Also validates divisibility constraints:
        - itot must be divisible by nprocx
        - jtot must be divisible by nprocy
        - ktot must be divisible by nprocy
        """
        namoptions_path = (
            pathlib.Path(self.temp_dir) / f"namoptions.{self.experiment_name}"
        )

        if not namoptions_path.exists():
            logger.warning(
                f"namoptions.{self.experiment_name} not found, skipping NCPU validation"
            )
            return

        # Read namoptions to get nprocx, nprocy, itot, jtot, ktot
        nprocx = None
        nprocy = None
        itot = None
        jtot = None
        ktot = None

        with open(namoptions_path, "r") as f:
            in_run = False
            in_domain = False
            for line in f:
                stripped = line.strip()
                if stripped.startswith("&RUN"):
                    in_run = True
                elif stripped.startswith("&DOMAIN"):
                    in_domain = True
                elif stripped.startswith("/"):
                    in_run = False
                    in_domain = False
                elif in_run and "nprocx" in stripped and "=" in stripped:
                    try:
                        nprocx = int(stripped.split("=")[1].strip().rstrip("."))
                    except (ValueError, IndexError):
                        pass
                elif in_run and "nprocy" in stripped and "=" in stripped:
                    try:
                        nprocy = int(stripped.split("=")[1].strip().rstrip("."))
                    except (ValueError, IndexError):
                        pass
                elif in_domain and "itot" in stripped and "=" in stripped:
                    try:
                        itot = int(stripped.split("=")[1].strip().rstrip("."))
                    except (ValueError, IndexError):
                        pass
                elif in_domain and "jtot" in stripped and "=" in stripped:
                    try:
                        jtot = int(stripped.split("=")[1].strip().rstrip("."))
                    except (ValueError, IndexError):
                        pass
                elif in_domain and "ktot" in stripped and "=" in stripped:
                    try:
                        ktot = int(stripped.split("=")[1].strip().rstrip("."))
                    except (ValueError, IndexError):
                        pass

        # Validate and sync
        if nprocx is not None and nprocy is not None:
            expected_ncpu = nprocx * nprocy

            # Check divisibility constraints
            if itot is not None and itot % nprocx != 0:
                raise ValueError(
                    f"itot ({itot}) must be divisible by nprocx ({nprocx}). "
                    f"Please adjust nprocx or itot in namoptions.{self.experiment_name}"
                )

            if jtot is not None and jtot % nprocy != 0:
                raise ValueError(
                    f"jtot ({jtot}) must be divisible by nprocy ({nprocy}). "
                    f"Please adjust nprocy or jtot in namoptions.{self.experiment_name}"
                )

            if ktot is not None and ktot % nprocy != 0:
                raise ValueError(
                    f"ktot ({ktot}) must be divisible by nprocy ({nprocy}). "
                    f"Please adjust nprocy or ktot in namoptions.{self.experiment_name}"
                )

            # Check if NCPU matches nprocx * nprocy
            if self.ncpu != expected_ncpu:
                logger.warning(
                    f"NCPU ({self.ncpu}) does not match nprocx * nprocy ({nprocx} * {nprocy} = {expected_ncpu}). "
                    f"Updating NCPU to {expected_ncpu} to match namoptions."
                )
                self.ncpu = expected_ncpu
        else:
            logger.warning(
                f"Could not read nprocx and/or nprocy from namoptions.{self.experiment_name}. "
                f"Using NCPU={self.ncpu} as specified."
            )

    def _create_config_sh(self) -> None:
        """Create a config.sh file where the environment variables are set."""
        # DA_EXPDIR should point to the parent directory containing experiments,
        # not the specific experiment directory, because MATLAB appends expnr to it

        config_sh_path = pathlib.Path(self.temp_dir) / "config.sh"
        matlab_bin_dir = pathlib.Path(self.matlab_bin).parent
        # Set DA_EXPDIR to parent directory so MATLAB can append expnr

        self.udales_root_path = pathlib.Path(self.udales_root_path)  # type: ignore[arg-type]
        da_expdir = self.temp_dir.parent
        with open(config_sh_path, "w") as f:
            f.write(f"export DA_EXPDIR={str(da_expdir)}\n")
            f.write(
                f"export DA_TOOLSDIR={str(self.udales_root_path.joinpath('tools'))}\n"
            )
            f.write(
                f"export DA_BUILD={str(self.udales_root_path.joinpath('build', 'release', 'u-dales'))}\n"
            )
            f.write(f"export DA_WORKDIR={str(self.work_dir)}\n")
            f.write(f"export NCPU={self.ncpu}\n")
            f.write(f"export MATLAB_BIN={str(self.matlab_bin)}\n")
            f.write(f"export PATH={matlab_bin_dir}:{os.environ.get('PATH', '')}\n")

    def _clean_work_dir(self, except_for_files: list[str] = []) -> None:
        """Delete the work directory contents (files and subdirectories)."""
        work_experiment_dir = self.work_dir.joinpath(self.experiment_name)
        if not work_experiment_dir.exists():
            return

        for item in work_experiment_dir.iterdir():
            if item.name in except_for_files:
                continue
            elif item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item)

    def _clean_temp_dir(self) -> None:
        """Clean the temp directory."""
        # Empty the temp directory by removing all its contents
        for item in pathlib.Path(self.temp_dir).iterdir():
            name = item.name
            lower_name = name.lower()
            # Exclude config.sh, any namoptions*, and any *.stl (case-insensitive)
            if lower_name == "config.sh":
                continue
            if lower_name.startswith("namoptions"):
                continue
            if lower_name.endswith(".stl"):
                continue
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item)

    def run_preprocessing(self, python_or_matlab: str = "python") -> None:
        """Run preprocessing."""

        logger.info("Running preprocessing...")

        self._clean_temp_dir()
        self._clean_work_dir()

        if python_or_matlab == "python":
            # Use Python-based preprocessing script
            script_path = pathlib.Path(__file__).parent.parent.parent / "shell_scripts" / "write_inputs.sh"
            
            command = [
                "bash",
                str(script_path),
                str(self.temp_dir),
            ]
            env = os.environ.copy()
            # Set environment variables needed by the script
            env["DA_EXPDIR"] = str(self.temp_dir.parent)
            env["DA_TOOLSDIR"] = str(self.udales_root_path.joinpath("tools"))  # type: ignore[union-attr]
            
        elif python_or_matlab == "matlab":
            # Use MATLAB-based preprocessing script
            command = [
                "bash",
                str(self.udales_root_path.joinpath("tools", "write_inputs.sh")),  # type: ignore[union-attr]
                str(self.temp_dir),
            ]
            # Add MATLAB bin directory to PATH so the script can find 'matlab'
            env = os.environ.copy()
            matlab_bin_dir = str(pathlib.Path(self.matlab_bin).parent)
            env["PATH"] = f"{matlab_bin_dir}:{env.get('PATH', '')}"
            logger.info("Running preprocessing...")
            subprocess.run(command, check=True, env=env)

            time.sleep(90)  # Wait for preprocessing to complete
        
        subprocess.run(
            command, check=True, env=env, stdout=self.stdout, stderr=self.stderr
        )

        logger.info("Preprocessing completed.")

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset | None:
        """Run the forward model."""

        if params is not None:
            if "inflow_angle" in params:
                self.inflow_angle = params.inflow_angle.item()
            if "velocity_magnitude" in params:
                self.velocity_magnitude = params.velocity_magnitude.item()
            if "pressure_gradient_magnitude" in params:
                self.pressure_gradient_magnitude = (
                    params.pressure_gradient_magnitude.item()
                )

        self._apply_inflow_settings(
            inflow_angle=self.inflow_angle,
            velocity_magnitude=self.velocity_magnitude,
            pressure_gradient_magnitude=self.pressure_gradient_magnitude,
        )

        logger.info("Running forward model...")
        command = [
            "bash",
            str(self.udales_root_path.joinpath("tools", "local_execute.sh")),  # type: ignore[union-attr]
            str(self.temp_dir),
        ]

        subprocess.run(command, check=True, stdout=self.stdout, stderr=self.stderr)

        state = xarray.open_dataset(
            self.work_dir.joinpath(
                self.experiment_name, f"fielddump.{self.experiment_name}.nc"
            ),
            engine="netcdf4",
        )

        self._clean_work_dir()

        return state
