"""Utilities for handling rollout forward model results."""

import pathlib
import shutil

import xarray
from pyudales.utils.dir_utils import DirectoryPaths


def collect_rollout_results(
    sim_name: str,
    rollout_step: int,
    dirs: DirectoryPaths,
) -> xarray.Dataset:
    """Collect the results from the rollout forward model.

    If sim_name.nc already exists, load it and concatenate with the rollout file
    along the time dimension. If sim_name.nc does not exist, rename the rollout
    file to sim_name.nc.

    Args:
        sim_name: Base name for the simulation results file.
        rollout_step: The rollout step number.
        dirs: Directory paths.

    Returns:
        The collected dataset, either concatenated or renamed.
    """
    sim_file = dirs.results_dir / f"{sim_name}.nc"  # type: ignore[operator]
    rollout_file = dirs.results_dir / f"{sim_name}_rollout_{rollout_step}.nc"  # type: ignore[operator]

    if sim_file.exists():
        # Load both datasets into memory and close file handles before writing
        # (cannot write to sim_file while it is still open for reading)
        existing_data = xarray.open_dataset(sim_file, engine="netcdf4")
        rollout_data = xarray.open_dataset(rollout_file, engine="netcdf4")

        combined_state = xarray.concat(
            [existing_data, rollout_data], dim="time", join="override"
        )
        rollout_file.unlink(missing_ok=True)
        sim_file.unlink(missing_ok=True)
        combined_state.to_netcdf(sim_file)
    else:
        # Rename rollout file to sim_name.nc
        shutil.move(str(rollout_file), str(sim_file))
