"""Real-PALM check of a periodic, nudging-driven warm start.

A tiny periodic cold window, then a 0.4 s warm window from its last frame: the
first warm output must still look like the injected u/v, not like them shifted
one cell (the LOD=2 reader puts xu/yv element i at index i+1; see
docs/pypalm.md §8 "Warm-start grid contract").
"""

import pathlib

import numpy as np
import pytest
import xarray
from hydra.utils import instantiate

from tests.conftest import compose


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@pytest.mark.integration  # type: ignore[misc]
def test_periodic_warm_start_keeps_u_v_in_place(tmp_path: pathlib.Path) -> None:
    cfg = compose(
        "forward",
        "+test=forward",
        "model=pypalm_tiny",
        "model.forward_model.boundary_condition=periodic",
        "model.forward_model.inlet_turbulence.enabled=false",
        "model.forward_model.inlet_turbulence.initial_seed=true",
        "model.forward_model.nudging_config.nnudge_meters=4.0",
        "time.simulation_time=6.0",
        "time.spinup_time=0.0",
        root=tmp_path,
    )
    params = xarray.Dataset({"inflow_angle": 10.0, "velocity_magnitude": 5.0})
    cold = instantiate(cfg.model.forward_model)(params=params)
    assert cold is not None
    state = cold.isel(time=[-1])

    warm_model = instantiate(
        cfg.model.forward_model, simulation_time=0.4, output_frequency=0.2
    )
    warm = warm_model(params=params, state=state)
    assert warm is not None
    first = warm.isel(time=0)

    # Column x = 0 (row y = 0) is not in the driver; PALM copies its neighbour.
    u_old = state.u.isel(time=0).values[..., 1:]
    u_new = first.u.values[..., 1:]
    v_old = state.v.isel(time=0).values[:, 1:, :]
    v_new = first.v.values[:, 1:, :]
    assert _corr(u_new, u_old) > _corr(u_new, np.roll(u_old, 1, axis=-1))
    assert _corr(v_new, v_old) > _corr(v_new, np.roll(v_old, 1, axis=-2))
    assert _corr(u_new, u_old) > 0.95
