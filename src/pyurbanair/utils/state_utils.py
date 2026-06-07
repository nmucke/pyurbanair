import numpy as np
import xarray


def get_velocity_magnitude_field(state: xarray.Dataset) -> np.ndarray:
    """Get the velocity magnitude field from a state."""
    u = state.u.values
    v = state.v.values
    w = state.w.values
    return np.sqrt(u**2 + v**2 + w**2)
