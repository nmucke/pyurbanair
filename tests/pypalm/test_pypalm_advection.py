"""Advection scheme of the xie_and_castro case: ws-scheme, no grid-scale streaks.

With the non-dissipative pw-scheme, 2-cell wiggles from the building edges
travel upstream (the group velocity of a 2-dx wave is -U under central
differences) and stand in front of the array as spanwise streaks, even with a
laminar, y-homogeneous inflow. See docs/pypalm.md §8 "Grid-scale streaks".
"""

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray
from hydra.utils import instantiate

from tests.conftest import compose

_CASE = ["case=xie_and_castro", "model=pypalm"]


def _model(tmp_path: pathlib.Path, *overrides: str) -> Any:
    cfg = compose("forward", "+test=forward", *_CASE, *overrides, root=tmp_path)
    return instantiate(cfg.model.forward_model)


@pytest.mark.parametrize("bc", ["inflow_outflow", "periodic"])  # type: ignore[misc]
def test_case_stages_the_ws_scheme(tmp_path: pathlib.Path, bc: str) -> None:
    from pypalm.utils.p3d_utils import P3DFile

    overrides = [f"flow.boundary_condition={bc}", "model.forward_model.ncpu=4"]
    if bc == "periodic":
        overrides.append("flow.inlet_turbulence.enabled=false")
    p3d = P3DFile(_model(tmp_path, *overrides).p3d_path)
    for key in ("momentum_advec", "scalar_advec"):
        assert p3d.get_value("initialization_parameters", key) == "'ws-scheme'"


@pytest.mark.integration  # type: ignore[misc]
def test_laminar_inflow_stays_y_homogeneous_upstream(tmp_path: pathlib.Path) -> None:
    """A y-homogeneous laminar inflow keeps a y-homogeneous mean before the array.

    Measured on this grid (60 s after 30 s spin-up): the y-std of the
    time-mean u, 4-12 m behind the face, is 0.10-0.34 m/s with the pw-scheme
    and 0.003-0.013 m/s with the ws-scheme.
    """
    fm = _model(
        tmp_path,
        "model.forward_model.ncpu=1",
        "domain.nx=30",
        "domain.ny=20",
        "domain.nz=16",
        "domain.bounds=[[-20.0,40.0],[0.0,40.0],[0.0,32.0]]",
        "time.simulation_time=60.0",
        "time.output_frequency=1.0",
        "time.spinup_time=30.0",
        "flow.boundary_condition=inflow_outflow",
        "flow.inlet_turbulence.enabled=false",
        "model.forward_model.inlet_turbulence.initial_seed=false",
    )
    state = fm.run_single(
        None, xarray.Dataset({"inflow_angle": 10.0, "velocity_magnitude": 5.0})
    )
    upstream = (
        state.u.mean("time").sel(xu=slice(-16.0, -10.0), z=slice(4.0, 16.0)).load()
    )
    assert np.isfinite(upstream.values).all()
    assert float(upstream.std("y").max()) < 0.05
