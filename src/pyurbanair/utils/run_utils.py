from typing import Any

import numpy as np
import xarray


def add_velocity_magnitude(state: xarray.Dataset) -> xarray.Dataset:
    if not all(v in state.data_vars for v in ("u", "v", "w")):
        return state
    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    return state.assign(vel_magnitude=(state["u"].dims, vel_magnitude))


def extract_2d_slice(data_array: xarray.DataArray) -> np.ndarray:
    da = data_array
    if "time" in da.dims:
        da = da.isel(time=-1)
    for z_dim in ("z", "zm", "zt"):
        if z_dim in da.dims:
            da = da.isel({z_dim: len(da[z_dim]) // 2})
            break
    if da.ndim > 2:
        indexers = {dim: 0 for dim in da.dims[:-2]}
        da = da.isel(indexers)
    return np.asarray(da.values)


def get_ensemble_mean_field(
    output: tuple[xarray.Dataset, xarray.Dataset] | xarray.Dataset | None,
    esmda: Any,
    num_esmda_steps: int,
    ensemble_size: int,
) -> tuple[xarray.Dataset, xarray.Dataset]:
    """Get ensemble-mean state history and parameter output from ESMDA output."""
    if output is None:
        raise ValueError("ESMDA output is None.")

    if isinstance(output, tuple):
        params = output[0]
        ensemble_mean_field = output[1]
        if "ensemble" in ensemble_mean_field.dims:
            ensemble_mean_field = ensemble_mean_field.mean(dim="ensemble")
    else:
        params = output
        ensemble_mean_steps = []
        for i in range(num_esmda_steps + 1):
            esmda_step = esmda.get_state(step=i, ensemble_member=0)
            for j in range(1, ensemble_size):
                esmda_state = esmda.get_state(step=i, ensemble_member=j)
                for var in esmda_step.data_vars:
                    esmda_step[var].values = (
                        esmda_step[var].values + esmda_state[var].values
                    )
            for var in esmda_step.data_vars:
                esmda_step[var].values /= ensemble_size
            ensemble_mean_steps.append(esmda_step)
        ensemble_mean_field = xarray.concat(
            ensemble_mean_steps,
            dim="esmda_step",
            join="override",
        )

    return ensemble_mean_field, params
