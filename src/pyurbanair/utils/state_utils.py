import numpy as np
import xarray


def get_velocity_magnitude_field(state: xarray.Dataset) -> np.ndarray:
    """Get the velocity magnitude field from a state."""
    u = state.u.values
    v = state.v.values
    w = state.w.values
    return np.sqrt(u**2 + v**2 + w**2)


def get_ensemble_mean_field(states: list[xarray.Dataset]) -> xarray.Dataset:
    """Get the ensemble mean field from a list of states."""
    u = np.stack([state.u.values for state in states], axis=0)
    v = np.stack([state.v.values for state in states], axis=0)
    w = np.stack([state.w.values for state in states], axis=0)
    return xarray.Dataset(
        data_vars={
            "u": (["time", "z", "y", "x"], np.mean(u, axis=0)),
            "v": (["time", "z", "y", "x"], np.mean(v, axis=0)),
            "w": (["time", "z", "y", "x"], np.mean(w, axis=0)),
        },
        coords={
            "time": [0],
            "zt": states[0].zt.values,
            "yt": states[0].yt.values,
            "xt": states[0].xt.values,
            "zm": states[0].zm.values,
            "ym": states[0].ym.values,
            "xm": states[0].xm.values,
        },
    )
