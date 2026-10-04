import numpy as np
import xarray


def add_velocity_magnitude(state: xarray.Dataset) -> xarray.Dataset:
    if not all(v in state.data_vars for v in ("u", "v", "w")):
        return state
    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    return state.assign(vel_magnitude=(state["u"].dims, vel_magnitude))
