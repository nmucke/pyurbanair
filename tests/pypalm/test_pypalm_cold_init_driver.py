"""Staging tests for the inflow_outflow cold-start LOD=1 init driver.

Like ``test_pypalm_outflow_damping.py`` these compose a real smoke model and
check the staged files WITHOUT running PALM (``run`` is stubbed out of
``run_single``). Under inflow_outflow a cold start writes the inflow profile as
LOD=1 ``init_atmosphere_*`` fields into ``<name>_dynamic`` and sets
``initializing_actions='read_from_file'``, because ``set_constant_profiles``
zeroes u/v on the first fluid level, inflow face included (docs/pypalm.md §8).
Periodic cold starts and warm starts keep their own init paths.
"""

import pathlib
from typing import Any

import numpy as np
import xarray
from hydra.utils import instantiate

from tests.conftest import compose

NX, NY, NZ = 20, 20, 4
_SMOKE = [
    f"domain.nx={NX}",
    f"domain.ny={NY}",
    f"domain.nz={NZ}",
    "domain.bounds=[[0.0,20.0],[0.0,20.0],[0.0,10.0]]",
    "time.simulation_time=3.0",
    "time.output_frequency=1.0",
    "time.spinup_time=3.0",
    "model=pypalm",
    "model.forward_model.inlet_turbulence.enabled=false",
    "model.forward_model.inlet_turbulence.initial_seed=true",
    "model.forward_model.nudging_config.profile_config.type=power_law",
    "model.forward_model.nudging_config.profile_config.alpha=0.25",
    # Reference height = domain top, as _expected_profile assumes.
    "++model.forward_model.nudging_config.profile_config.z_ref=null",
    "model.forward_model.nudging_config.nnudge_meters=4.0",
]

# inflow_outflow selects the multigrid solver: ncpu must divide nx.
_INFLOW_OUTFLOW = [
    "model.forward_model.boundary_condition=inflow_outflow",
    "model.forward_model.ncpu=4",
]
_PERIODIC = ["model.forward_model.boundary_condition=periodic"]

_INIT_VARS = ("u", "v", "w", "pt", "qv")


def _make_model(tmp_path: pathlib.Path, *extra_overrides: str) -> Any:
    cfg = compose("forward", "+test=forward", *_SMOKE, *extra_overrides, root=tmp_path)
    fm = instantiate(cfg.model.forward_model)
    # Stage only: no PALM run, no output to read back.
    fm.run = lambda: None
    fm._load_and_postprocess_state = lambda: None
    return fm


def _static_params(angle: float = 30.0, speed: float = 5.0) -> xarray.Dataset:
    return xarray.Dataset(
        data_vars={"inflow_angle": float(angle), "velocity_magnitude": float(speed)}
    )


def _time_varying_params() -> xarray.Dataset:
    return xarray.Dataset(
        data_vars={
            "inflow_angle": ("time", np.array([0.0, 30.0, 60.0])),
            "velocity_magnitude": 5.0,
        },
        coords={"time": np.array([0.0, 50.0, 100.0])},
    )


def _warm_state(fm: Any) -> xarray.Dataset:
    """A constant postprocessed-style state on the model's grid."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = fm.bounds
    dx, dy, dz = (xmax - xmin) / NX, (ymax - ymin) / NY, (zmax - zmin) / NZ
    coords = {
        "time": [0],
        "x": (np.arange(NX) + 0.5) * dx + xmin,
        "xu": np.arange(NX - 1) * dx + xmin,
        "y": (np.arange(NY) + 0.5) * dy + ymin,
        "yv": np.arange(NY - 1) * dy + ymin,
        "z": (np.arange(NZ) + 0.5) * dz + zmin,
    }
    return xarray.Dataset(
        data_vars={
            "u": (("time", "z", "y", "xu"), np.full((1, NZ, NY, NX - 1), 3.0)),
            "v": (("time", "z", "yv", "x"), np.full((1, NZ, NY - 1, NX), -1.0)),
            "w": (("time", "z", "y", "x"), np.zeros((1, NZ, NY, NX))),
        },
        coords=coords,
    )


def _init_actions(fm: Any) -> str:
    from pypalm.utils.p3d_utils import P3DFile

    value = P3DFile(fm.p3d_path).get_value(
        "initialization_parameters", "initializing_actions"
    )
    return str(value).strip("'\"")


def _expected_profile(fm: Any, angle: float = 30.0, speed: float = 5.0) -> Any:
    from pypalm.utils.inflow_utils import angle_to_velocity
    from pypalm.utils.vertical_profile import build_profile_shape

    zmin, zmax = fm.bounds[2]
    dz = (zmax - zmin) / NZ
    heights = np.arange(NZ) * dz + 0.5 * dz + zmin
    shape = build_profile_shape(
        {"type": "power_law", "alpha": 0.25}, heights=heights, zsize=zmax - zmin
    )
    u0, v0 = angle_to_velocity(angle, speed)
    return shape * float(u0), shape * float(v0)


def test_inflow_outflow_cold_start_writes_lod1_driver(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    fm.run_single(params=_static_params())

    assert _init_actions(fm) == "read_from_file"
    ds = xarray.open_dataset(fm.dynamic_driver_path)
    for name in _INIT_VARS:
        var = ds[f"init_atmosphere_{name}"]
        assert int(var.attrs["lod"]) == 1, name
        assert var.dims == (("zw",) if name == "w" else ("z",)), name
    # LOD=1 still needs the horizontal dims (DRV0003), at PALM's lengths.
    assert {k: ds.sizes[k] for k in ("x", "xu", "y", "yv", "z", "zw")} == {
        "x": NX,
        "xu": NX - 1,
        "y": NY,
        "yv": NY - 1,
        "z": NZ,
        "zw": NZ - 1,
    }
    # 0-based value-checked vertical axis (DRV0005).
    assert np.allclose(ds["z"].values, (np.arange(NZ) + 0.5) * 2.5)

    # Same cell-centre profile as u_profile/v_profile (which are kept).
    u_exp, v_exp = _expected_profile(fm)
    assert np.allclose(ds["init_atmosphere_u"].values, u_exp, atol=1e-5)
    assert np.allclose(ds["init_atmosphere_v"].values, v_exp, atol=1e-5)
    assert np.all(ds["init_atmosphere_w"].values == 0.0)
    assert np.allclose(ds["init_atmosphere_pt"].values, 300.0)
    text = fm.p3d_path.read_text()
    assert "u_profile" in text and "uv_heights" in text


def test_periodic_cold_start_unchanged(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_PERIODIC)
    fm.run_single(params=_static_params())

    assert _init_actions(fm) == "set_constant_profiles"
    assert not fm.dynamic_driver_path.exists()


def test_warm_start_keeps_lod2_and_drops_cold_driver(tmp_path: pathlib.Path) -> None:
    """A warm window after a cold one in the same dir is pure LOD=2."""
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    fm.run_single(params=_static_params())
    assert (
        int(xarray.open_dataset(fm.dynamic_driver_path)["init_atmosphere_u"].lod) == 1
    )

    fm.run_single(state=_warm_state(fm), params=_static_params())
    assert _init_actions(fm) == "read_from_file"
    ds = xarray.open_dataset(fm.dynamic_driver_path)
    for name in _INIT_VARS:
        assert int(ds[f"init_atmosphere_{name}"].attrs["lod"]) == 2, name
    assert ds["init_atmosphere_u"].dims == ("z", "y", "xu")


def test_cold_start_after_warm_start_rewrites_lod1(tmp_path: pathlib.Path) -> None:
    """A stale LOD=2 driver from a reused experiment dir does not leak."""
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    fm.run_single(state=_warm_state(fm), params=_static_params())
    fm.run_single(params=_static_params())

    ds = xarray.open_dataset(fm.dynamic_driver_path)
    assert ds["init_atmosphere_u"].dims == ("z",)
    assert int(ds["init_atmosphere_u"].attrs["lod"]) == 1


def test_time_varying_driver_merges_inflow_planes_and_lod1(
    tmp_path: pathlib.Path,
) -> None:
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    fm.run_single(params=_time_varying_params())

    assert _init_actions(fm) == "read_from_file"
    ds = xarray.open_dataset(fm.dynamic_driver_path)
    for name in ("u", "v", "w", "e", "pt"):
        assert f"inflow_plane_{name}" in ds.data_vars, name
    for name in _INIT_VARS:
        assert int(ds[f"init_atmosphere_{name}"].attrs["lod"]) == 1, name
    # The init profile is the t=0 inflow (angle 0, speed 5).
    u_exp, v_exp = _expected_profile(fm, angle=0.0)
    assert np.allclose(ds["init_atmosphere_u"].values, u_exp, atol=1e-5)
    assert np.allclose(ds["init_atmosphere_v"].values, v_exp, atol=1e-5)
    assert ds.sizes["time_inflow"] == 4  # spinup plateau + 3 schedule points


def test_warm_start_static_imposes_inflow_planes(tmp_path: pathlib.Path) -> None:
    """A static warm start keeps the inflow face on the prescribed profile.

    The LOD=2 reader leaves the inflow ghost column u(nxl-1) at 0, and PALM
    copies it onto the face every step (dynamics_mod.f90:1287), so without the
    inflow planes the face sat at u=0 for the whole warm window. The constant
    inflow-plane driver re-imposes it every step (turbulent_inflow_rff_bc).
    """
    from pypalm.utils.p3d_utils import P3DFile

    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    fm.run_single(params=_static_params())
    fm.run_single(state=_warm_state(fm), params=_static_params())

    p3d = P3DFile(fm.p3d_path)
    section = "turbulent_inflow_parameters"
    assert p3d.get_value(section, "switch_off_module") == ".false."
    assert p3d.get_value(section, "turbulent_inflow_method").strip("'\"") == (
        "read_from_file"
    )
    ds = xarray.open_dataset(fm.dynamic_driver_path)
    assert int(ds["init_atmosphere_u"].attrs["lod"]) == 2
    # Constant in time and y, covering the whole (spin-up free) warm window.
    assert ds["time_inflow"].values[0] == 0.0
    assert ds["time_inflow"].values[-1] >= fm.simulation_time
    u_exp, v_exp = _expected_profile(fm)
    plane_u = ds["inflow_plane_u"].values  # (time_inflow, z, y)
    plane_v = ds["inflow_plane_v"].values
    assert np.allclose(plane_u, u_exp[None, :, None], atol=1e-5)
    assert np.allclose(plane_v, v_exp[None, :, None], atol=1e-5)
    assert np.all(ds["inflow_plane_w"].values == 0.0)


def test_cold_static_uses_the_same_inflow_planes(tmp_path: pathlib.Path) -> None:
    """Static inflow_outflow cold starts share the driver path of warm starts."""
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    fm.run_single(params=_static_params())

    ds = xarray.open_dataset(fm.dynamic_driver_path)
    u_exp, _ = _expected_profile(fm)
    assert np.allclose(ds["inflow_plane_u"].values, u_exp[None, :, None], atol=1e-5)
    # Spin-up plateau + the two knots of the constant schedule.
    assert list(ds["time_inflow"].values) == [0.0, 3.0, 6.0]
