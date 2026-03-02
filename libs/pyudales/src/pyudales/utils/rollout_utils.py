"""Utilities for handling rollout forward model results."""

import pathlib
import shutil

import xarray


def collect_rollout_results(
    sim_name: str,
    rollout_step: int,
    results_dir: pathlib.Path,
) -> None:
    """Collect the results from the rollout forward model.

    If sim_name.nc already exists, load it and concatenate with the rollout file
    along the time dimension. If sim_name.nc does not exist, rename the rollout
    file to sim_name.nc.

    Args:
        sim_name: Base name for the simulation results file.
        rollout_step: The rollout step number.
        results_dir: Directory where results are stored.
    """
    sim_file = results_dir / f"{sim_name}.nc"
    rollout_file = results_dir / f"{sim_name}_rollout_{rollout_step}.nc"

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
