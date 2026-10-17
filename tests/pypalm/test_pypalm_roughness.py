"""The ``roughness_length`` knob: wall z0 written into PALM's ``_p3d``.

The staging tests compose a real model but do not run PALM; the integration
test runs a tiny laminar case at two z0 values. See docs/pypalm.md "Roughness".
"""

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray
from hydra.utils import instantiate

from tests.conftest import compose

# dz = 10 m / 4 = 2.5 m, so PALM accepts z0 < 1.25 m.
_SMOKE = [
    "domain.nx=20",
    "domain.ny=20",
    "domain.nz=4",
    "domain.bounds=[[0.0,20.0],[0.0,20.0],[0.0,10.0]]",
    "time.simulation_time=3.0",
    "time.output_frequency=1.0",
    "time.spinup_time=3.0",
    "model=pypalm",
    "model.forward_model.boundary_condition=inflow_outflow",
    "model.forward_model.ncpu=4",
    "model.forward_model.inlet_turbulence.enabled=false",
]
PARAMS = {"inflow_angle": 0.0, "velocity_magnitude": 5.0}


def _make_model(tmp_path: pathlib.Path, *overrides: str, drop: bool = False) -> Any:
    cfg = compose("forward", "+test=forward", *_SMOKE, *overrides, root=tmp_path)
    if drop:
        # The model as built before the knob existed.
        del cfg.model.forward_model.roughness_length
    return instantiate(cfg.model.forward_model)


def _stage(fm: Any, **params: float) -> str:
    fm._apply_inflow_settings(xarray.Dataset({**PARAMS, **params}))
    return str(fm.p3d_path.read_text())


def _z0(fm: Any) -> str:
    from pypalm.utils.p3d_utils import P3DFile

    value = P3DFile(fm.p3d_path).get_value(
        "initialization_parameters", "roughness_length"
    )
    assert value is not None
    return value


def test_default_is_byte_identical(tmp_path: pathlib.Path) -> None:
    """flow.roughness_length's 0.05 is the case's own value: nothing changes."""
    absent = _make_model(tmp_path / "absent", drop=True)
    default = _make_model(tmp_path / "default")
    assert absent.roughness_length is None and default.roughness_length == 0.05
    assert _stage(default) == _stage(absent)
    assert _z0(default) == "0.05"


def test_params_value_is_written(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path)
    text = _stage(fm, roughness_length=0.02)
    assert _z0(fm) == "0.02"
    assert text.count("roughness_length") == 1


def test_model_config_value_is_written(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, "flow.roughness_length=0.3")
    _stage(fm)
    assert _z0(fm) == "0.3"


def test_ensemble_members_get_their_own_value(tmp_path: pathlib.Path) -> None:
    from pypalm.utils.forward_model_utils import create_new_forward_model

    base = _make_model(tmp_path)
    base_text = base.p3d_path.read_text()
    members = [
        create_new_forward_model(base, tmp_path / "ensemble", f"member_{i}")
        for i in range(2)
    ]
    for member, z0 in zip(members, (0.01, 0.2)):
        _stage(member, roughness_length=z0)
    assert [_z0(m) for m in members] == ["0.01", "0.2"]
    assert base.p3d_path.read_text() == base_text


def test_warm_start_reapplies(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm window without params re-applies the member's z0."""
    fm = _make_model(tmp_path)
    _stage(fm, roughness_length=0.07)
    # Put the template's value back, as a fresh _p3d would have it.
    from pypalm.utils.p3d_utils import P3DFile

    p3d = P3DFile(fm.p3d_path)
    p3d.set_value("initialization_parameters", "roughness_length", 0.05)
    p3d.write()

    monkeypatch.setattr(fm, "_apply_warmstart", lambda state: None)
    monkeypatch.setattr(fm, "run", lambda: None)
    monkeypatch.setattr(fm, "_load_and_postprocess_state", xarray.Dataset)
    fm.run_single(state=xarray.Dataset(), params=None)
    assert _z0(fm) == "0.07"


@pytest.mark.parametrize("value", [0.0, -0.1, 1.25, 2.0, np.nan])  # type: ignore[misc]
def test_out_of_range_raises(tmp_path: pathlib.Path, value: float) -> None:
    fm = _make_model(tmp_path)
    with pytest.raises(ValueError, match="0 < z0 < 0.5\\*dz"):
        _stage(fm, roughness_length=value)


def test_sgs_constant_conflict_raises(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path)
    with pytest.raises(ValueError, match="constant_flux_layer"):
        _stage(fm, roughness_length=0.1, sgs_constant=0.5)

    fm = _make_model(tmp_path / "config", "model.forward_model.sgs_constant=0.5")
    with pytest.raises(ValueError, match="constant_flux_layer"):
        _stage(fm, roughness_length=0.1)


def test_sgs_constant_with_the_default_z0_still_runs(tmp_path: pathlib.Path) -> None:
    """The default z0 writes nothing, so it does not block a km_constant run."""
    fm = _make_model(tmp_path)
    _stage(fm, sgs_constant=0.5)
    assert _z0(fm) == "0.05"


def test_sgs_constant_skips_a_config_z0_unlike_the_template(
    tmp_path: pathlib.Path,
) -> None:
    """A template with another z0 (barcelona, urbantales ship 0.1) must not make
    a km_constant run raise: the config value is inert there, so it is skipped."""
    fm = _make_model(tmp_path, "flow.roughness_length=0.3")
    _stage(fm, sgs_constant=0.5)
    assert _z0(fm) == "0.05"


@pytest.mark.integration  # type: ignore[misc]
def test_e2e_rougher_surface_slows_the_near_wall_flow(tmp_path: pathlib.Path) -> None:
    """A flat-floor laminar run at two z0: higher z0, lower first-level speed."""
    speeds = {}
    for z0 in (0.01, 0.2):
        fm = _make_model(
            tmp_path / str(z0),
            "model.forward_model.ncpu=1",
            "domain.nx=20",
            "domain.ny=20",
            "domain.nz=16",
            "domain.bounds=[[-40.0,0.0],[0.0,40.0],[0.0,32.0]]",
            "time.simulation_time=20.0",
            "time.spinup_time=20.0",
            "time.output_frequency=5.0",
            "model.forward_model.inlet_turbulence.initial_seed=false",
        )
        state = fm.run_single(None, xarray.Dataset({**PARAMS, "roughness_length": z0}))
        assert _z0(fm) == str(z0)
        u = state.u.isel(z=0).mean("time").load()
        assert np.isfinite(u.values).all()
        speeds[z0] = float(u.mean())
    print(f"near-wall mean u by z0: {speeds}")
    assert speeds[0.2] < speeds[0.01]
