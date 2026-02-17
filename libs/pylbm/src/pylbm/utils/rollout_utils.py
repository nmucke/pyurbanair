"""Utilities for handling rollout result aggregation for LBM."""

import pathlib
import shutil

import xarray


def collect_rollout_results(
    sim_name: str,
    rollout_step: int,
    results_dir: pathlib.Path,
) -> None:
    """
    Aggregate per-step rollout output into one file on disk.

    If `{sim_name}.nc` exists, append `{sim_name}_rollout_{rollout_step}.nc` along
    the time dimension. Otherwise, rename the rollout file to `{sim_name}.nc`.
    """
    sim_file = results_dir / f"{sim_name}.nc"
    rollout_file = results_dir / f"{sim_name}_rollout_{rollout_step}.nc"

    if sim_file.exists():
        existing_data = xarray.open_dataset(sim_file, engine="netcdf4").load()
        rollout_data = xarray.open_dataset(rollout_file, engine="netcdf4").load()

        combined_state = xarray.concat(
            [existing_data, rollout_data], dim="time", join="override"
        )
        sim_file.unlink(missing_ok=True)
        rollout_file.unlink(missing_ok=True)
        combined_state.to_netcdf(sim_file)
    else:
        shutil.move(str(rollout_file), str(sim_file))
