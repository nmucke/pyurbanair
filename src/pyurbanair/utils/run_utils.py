from typing import Any

import numpy as np
import xarray


def add_velocity_magnitude(state: xarray.Dataset) -> xarray.Dataset:
    if not all(v in state.data_vars for v in ("u", "v", "w")):
        return state
    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    return state.assign(vel_magnitude=(state["u"].dims, vel_magnitude))


def extract_2d_slice(
    data_array: xarray.DataArray, z_level: int | None = None
) -> np.ndarray:
    da = data_array
    if "time" in da.dims:
        da = da.isel(time=-1)
    for z_dim in ("z", "zm", "zt"):
        if z_dim in da.dims:
            da = da.isel(
                {z_dim: z_level if z_level is not None else len(da[z_dim]) // 2}
            )
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
            member_states = [
                esmda.get_state(step=i, ensemble_member=j) for j in range(ensemble_size)
            ]

            # Some ensemble members can have slightly different time lengths.
            # Truncate to the shortest available length before averaging.
            if any("time" in state.dims for state in member_states):
                common_time_len = min(
                    state.sizes["time"]
                    for state in member_states
                    if "time" in state.dims
                )
                member_states = [
                    (
                        state.isel(time=slice(0, common_time_len))
                        if "time" in state.dims
                        else state
                    )
                    for state in member_states
                ]

            esmda_step = xarray.concat(
                member_states,
                dim="ensemble_member",
                join="override",
            ).mean(dim="ensemble_member")
            ensemble_mean_steps.append(esmda_step)

        if any("time" in step.dims for step in ensemble_mean_steps):
            common_time_len = min(
                step.sizes["time"]
                for step in ensemble_mean_steps
                if "time" in step.dims
            )
            ensemble_mean_steps = [
                (
                    step.isel(time=slice(0, common_time_len))
                    if "time" in step.dims
                    else step
                )
                for step in ensemble_mean_steps
            ]

        ensemble_mean_field = xarray.concat(
            ensemble_mean_steps,
            dim="esmda_step",
            join="override",
        )

    return ensemble_mean_field, params
