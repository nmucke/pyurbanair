"""Staging tests for ``inlet_turbulence.method: digital_filter`` (no PALM run).

The method writes turbulent ``inflow_plane_u/v/w`` into the ``_dynamic`` driver
and switches PALM's ``turbulent_inflow`` reader on. Covered here: the namelist
and file it stages, the plane statistics, the seed, window continuity through
the persisted clock, and that it reproduces pyudales' inlet planes. The solver
run is in ``test_digital_filter_inflow_runs_and_continues``.
"""

import json
import pathlib
from typing import Any

import numpy as np
import pytest
import xarray
from hydra.utils import instantiate

from tests.conftest import compose

_GRID = [
    "domain.nx=20",
    "domain.ny=20",
    "domain.nz=16",
    "domain.bounds=[[0.0,40.0],[0.0,40.0],[0.0,32.0]]",
    "time.simulation_time=20.0",
    "time.output_frequency=1.0",
    "time.spinup_time=10.0",
    "model=pypalm",
    "model.forward_model.boundary_condition=inflow_outflow",
    "model.forward_model.ncpu=1",
    "model.forward_model.random_seed=null",
    "+model.forward_model.nudging_config.profile_config.z_ref=64.0",
    "model.forward_model.nudging_config.profile_config.alpha=0.25",
    # The values the tests assert on.
    "model.forward_model.inlet_turbulence.enabled=true",
    "model.forward_model.inlet_turbulence.method=digital_filter",
    "model.forward_model.inlet_turbulence.initial_seed=false",
    "model.forward_model.inlet_turbulence.intensity=0.15",
    "model.forward_model.inlet_turbulence.length_scale_x=24.0",
    "model.forward_model.inlet_turbulence.length_scale_y=12.0",
    "model.forward_model.inlet_turbulence.length_scale_z=12.0",
    "model.forward_model.inlet_turbulence.time_step=0.5",
    "model.forward_model.inlet_turbulence.seed=null",
]
_OFF = ["model.forward_model.inlet_turbulence.enabled=false"]


def _model(tmp_path: pathlib.Path, *overrides: str) -> Any:
    cfg = compose("forward", "+test=forward", *_GRID, *overrides, root=tmp_path)
    return instantiate(cfg.model.forward_model)


def _params(angle: float = 10.0, speed: float = 6.0) -> xarray.Dataset:
    return xarray.Dataset({"inflow_angle": angle, "velocity_magnitude": speed})


def _planes(fm: Any) -> xarray.Dataset:
    return xarray.load_dataset(fm.dynamic_driver_path)


def _p3d(fm: Any, section: str, key: str) -> Any:
    from pypalm.utils.p3d_utils import P3DFile

    return P3DFile(fm.p3d_path).get_value(section, key)


def _shape_u(fm: Any) -> np.ndarray:
    z = (np.arange(16) + 0.5) * 2.0
    return np.asarray((z / 64.0) ** 0.25)


def test_stages_planes_and_switches_the_reader_on(tmp_path: pathlib.Path) -> None:
    fm = _model(tmp_path)
    fm._apply_inflow_settings(_params())

    assert _p3d(fm, "turbulent_inflow_parameters", "switch_off_module") == ".false."
    assert (
        _p3d(fm, "turbulent_inflow_parameters", "turbulent_inflow_method")
        == "'read_from_file'"
    )
    ds = _planes(fm)
    # 30 s window at 0.5 s, plus the record at t=0 and two margin records.
    assert ds.sizes["time_inflow"] == 63
    assert ds.time_inflow.values[0] == 0.0
    assert ds.inflow_plane_u.shape == (63, 16, 20)
    assert ds.inflow_plane_w.shape == (63, 15, 20)
    # PALM's random disturbances stay off.
    assert fm._apply_inlet_turbulence(warm_start=False) is False
    assert _p3d(fm, "runtime_parameters", "dt_disturb") is None


def test_plane_statistics_follow_the_config(tmp_path: pathlib.Path) -> None:
    # Short length scales: zeroing the plane mean of u' then costs ~2% of the
    # rms. At 12 m on this 40 x 32 m plane it shifts it by up to 30% (the same
    # in pyudales, see docs/pyudales.md 6.1).
    fm = _model(
        tmp_path,
        "time.simulation_time=600.0",
        "time.spinup_time=0.0",
        "model.forward_model.inlet_turbulence.length_scale_y=4.0",
        "model.forward_model.inlet_turbulence.length_scale_z=4.0",
    )
    fm._apply_inflow_settings(_params(angle=10.0, speed=6.0))
    ds = _planes(fm)
    u = ds.inflow_plane_u.values.astype(float)
    v = ds.inflow_plane_v.values.astype(float)
    shape = _shape_u(fm)
    u_mean = 6.0 * np.cos(np.deg2rad(10.0)) * shape
    v_mean = 6.0 * np.sin(np.deg2rad(10.0)) * shape

    # The plane mean of u' is zero per record: the bulk inflow is the mean one.
    plane_mean = u.mean(axis=(1, 2))
    bulk = u_mean.mean()
    assert np.allclose(plane_mean, bulk, rtol=1e-5)
    np.testing.assert_allclose(u.mean(axis=(0, 2)), u_mean, rtol=0.03)
    np.testing.assert_allclose(v.mean(axis=(0, 2)), v_mean, atol=0.05)
    # rms = intensity * |U(z)|, v' at 0.7x (mean removal costs a little).
    sigma = 0.15 * 6.0 * shape
    np.testing.assert_allclose(u.std(axis=(0, 2)), sigma, rtol=0.15)
    np.testing.assert_allclose(v.std(axis=(0, 2)), 0.7 * sigma, rtol=0.15)
    assert ds.inflow_plane_w.values.std() > 0.1


def test_matches_the_udales_inlet_planes(tmp_path: pathlib.Path) -> None:
    """Same seed and grid -> PALM gets uDALES' inlet realisation, value for value."""
    from pypalm.utils.inlet_turbulence_utils import build_digital_filter_planes
    from pyudales.utils.inlet_turbulence_utils import build_driver_planes

    times = np.arange(41) * 0.5
    angle = np.full(times.size, 10.0)
    speed = np.full(times.size, 6.0)
    profile = {"type": "power_law", "alpha": 0.25, "z_ref": 64.0}
    common = dict(
        times=times,
        start_index=7,
        inflow_angle=angle,
        velocity_magnitude=speed,
        profile_config=profile,
        intensity=0.15,
        length_scale_x=24.0,
        length_scale_y=12.0,
        length_scale_z=12.0,
        seed=1234,
    )
    u_ud, v_ud, w_ud = build_driver_planes(
        jtot=20, ktot=16, zsize=32.0, ylen=40.0, **common
    )
    u_pa, v_pa, w_pa = build_digital_filter_planes(
        bounds=((0.0, 40.0), (0.0, 40.0), (0.0, 32.0)), nz=16, ny=20, **common
    )
    # uDALES planes carry one halo cell on each side; w row k is face k-1.
    np.testing.assert_allclose(u_pa, u_ud[:, 1:17, 1:21], atol=1e-12)
    np.testing.assert_allclose(v_pa, v_ud[:, 1:17, 1:21], atol=1e-12)
    np.testing.assert_allclose(w_pa, w_ud[:, 2:17, 1:21], atol=1e-12)


def test_seed_selects_the_realisation(tmp_path: pathlib.Path) -> None:
    def planes(*overrides: str, root: str) -> np.ndarray:
        fm = _model(tmp_path / root, *overrides)
        fm._apply_inflow_settings(_params())
        return np.asarray(_planes(fm).inflow_plane_u.values)

    base = planes(root="a")
    np.testing.assert_array_equal(base, planes(root="b"))
    replica = planes("model.forward_model.random_seed=17", root="c")
    explicit = planes("model.forward_model.inlet_turbulence.seed=99", root="d")
    for other in (replica, explicit):
        assert not np.allclose(base, other)
        # Same statistics, different realisation.
        assert other.std() == pytest.approx(base.std(), rel=0.25)


def test_time_varying_params_set_the_plane_means(tmp_path: pathlib.Path) -> None:
    fm = _model(tmp_path, "model.forward_model.inlet_turbulence.intensity=0.0")
    params = xarray.Dataset(
        {
            "inflow_angle": ("time", [0.0, 0.0]),
            "velocity_magnitude": ("time", [4.0, 8.0]),
        },
        coords={"time": [0.0, 20.0]},
    )
    fm._apply_inflow_settings(params)
    ds = _planes(fm)
    top = ds.inflow_plane_u.isel(z=-1, y=0).values
    times = ds.time_inflow.values
    # 10 s spinup plateau at 4 m/s, then the ramp to 8 m/s at t = 30 s.
    expected = np.interp(times, [0.0, 10.0, 30.0], [4.0, 4.0, 8.0])
    np.testing.assert_allclose(top, expected * _shape_u(fm)[-1], rtol=1e-5)
    assert np.all(ds.inflow_plane_v.values == 0.0)


def _fake_run(fm: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fm, "run", lambda: None)
    monkeypatch.setattr(fm, "_load_and_postprocess_state", lambda: xarray.Dataset())
    monkeypatch.setattr(fm, "_apply_warmstart", lambda state: None)


def test_warm_window_continues_the_turbulence(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Window 2 starts where window 1's history left off, via the clock file."""
    long = _model(
        tmp_path / "long", "time.simulation_time=50.0", "time.spinup_time=10.0"
    )
    long._apply_inflow_settings(_params())
    reference = _planes(long).inflow_plane_u.values

    fm = _model(tmp_path / "windows")
    _fake_run(fm, monkeypatch)
    fm.run_single(None, _params())
    clock = json.loads(
        (fm.dirs.experiment_dir / "inlet_turbulence_clock.json").read_text()
    )
    assert clock["elapsed_time"] == pytest.approx(30.0)
    fm.run_single(xarray.Dataset(), _params())
    second = _planes(fm).inflow_plane_u.values
    # Window 2 (warm, no spinup) covers physical t = 30 .. 50 s.
    np.testing.assert_allclose(second[:41], reference[60:101], atol=1e-6)


def test_member_starts_its_own_clock(tmp_path: pathlib.Path) -> None:
    from pypalm.utils.forward_model_utils import create_new_forward_model

    fm = _model(tmp_path)
    clock = fm.dirs.experiment_dir / "inlet_turbulence_clock.json"
    clock.write_text(json.dumps({"elapsed_time": 30.0, "experiment_name": "x"}))
    member = create_new_forward_model(fm, tmp_path / "members", "member_1")
    assert member._elapsed_time == 0.0
    assert not (member.dirs.experiment_dir / clock.name).exists()


def test_disabled_writes_no_driver_and_no_clock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fm = _model(tmp_path, *_OFF)
    _fake_run(fm, monkeypatch)
    fm._apply_inflow_settings(_params())
    assert not fm.dynamic_driver_path.exists()
    assert _p3d(fm, "turbulent_inflow_parameters", "switch_off_module") is None
    fm.run_single(None, _params())
    assert not (fm.dirs.experiment_dir / "inlet_turbulence_clock.json").exists()


def test_bad_settings_raise(tmp_path: pathlib.Path) -> None:
    # Hydra wraps the constructor's ValueError.
    with pytest.raises(Exception, match="method"):
        _model(tmp_path / "a", "model.forward_model.inlet_turbulence.method=stg")
    with pytest.raises(Exception, match="length_scale_y"):
        _model(tmp_path / "b", "model.forward_model.inlet_turbulence.length_scale_y=0")
    fm = _model(tmp_path / "c", "model.forward_model.inlet_turbulence.time_step=0.7")
    with pytest.raises(ValueError, match="does not divide"):
        fm._apply_inflow_settings(_params())


@pytest.mark.integration  # type: ignore[misc]
def test_digital_filter_inflow_runs_and_continues(tmp_path: pathlib.Path) -> None:
    """Two tiny PALM windows: the inflow face carries the planes, warm one too."""
    cfg = compose(
        "forward",
        "+test=forward",
        "model=pypalm_tiny",
        "time.simulation_time=4.0",
        "time.spinup_time=2.0",
        "model.forward_model.boundary_condition=inflow_outflow",
        "model.forward_model.inlet_turbulence.enabled=true",
        "model.forward_model.inlet_turbulence.method=digital_filter",
        "model.forward_model.inlet_turbulence.initial_seed=false",
        "model.forward_model.inlet_turbulence.intensity=0.15",
        "model.forward_model.inlet_turbulence.time_step=0.5",
        "model.forward_model.inlet_turbulence.seed=7",
        root=tmp_path,
    )
    fm = instantiate(cfg.model.forward_model)
    first = fm.run_single(None, _params())
    planes = _planes(fm)
    second = fm.run_single(first.isel(time=-1), _params())

    for state in (first, second):
        assert np.isfinite(state.u.values).all()
        face = state.u.isel(xu=0)
        # Turbulent, not the constant mean profile -- including the warm window.
        assert float(face.std("y").mean()) > 0.01
    clock = json.loads(
        (fm.dirs.experiment_dir / "inlet_turbulence_clock.json").read_text()
    )
    assert clock["elapsed_time"] == pytest.approx(10.0)
    assert planes.sizes["time_inflow"] == 15
