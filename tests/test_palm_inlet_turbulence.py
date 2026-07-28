"""Staging tests for pypalm's switchable ``inlet_turbulence`` knob.

Like ``test_pypalm_nudging_driver.py`` these exercise a real, Hydra-composed
smoke model but WITHOUT running PALM — the wiring under test is pure namelist
staging: which ``_p3d`` keys flip per (inlet_turbulence × params × warm/cold).

The PALM-side mapping (and why it is NOT ``turbulent_inflow``) is documented in
``libs/pypalm/src/pypalm/utils/inlet_turbulence_utils.py`` and docs/pypalm.md §8.
"""

import pathlib
from typing import Any, Optional

import numpy as np
import pytest
import xarray
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

_SMOKE = [
    "domain.nx=20",
    "domain.ny=20",
    "domain.nz=4",
    "domain.bounds=[[0.0,20.0],[0.0,20.0],[0.0,10.0]]",
    "time.simulation_time=3.0",
    "time.output_frequency=1.0",
    "time.spinup_time=3.0",
    "model=pypalm",
]

# inflow_outflow selects PALM's multigrid pressure solver, which needs uniform
# slab subdomains — ncpu must divide the grid's x point count.
_INFLOW_OUTFLOW = [
    "model.forward_model.boundary_condition=inflow_outflow",
    "model.forward_model.ncpu=4",
]

_SEED_ON = "model.forward_model.inlet_turbulence.initial_seed=true"
_SEED_OFF = "model.forward_model.inlet_turbulence.initial_seed=false"
# Pin BOTH sub-switches everywhere: conf/model/pypalm.yaml tracks the active
# sweep, so inheriting either one silently changes what these tests assert.
_ON = ["model.forward_model.inlet_turbulence.enabled=true", _SEED_ON]
# Pin the knob rather than inheriting conf/model/pypalm.yaml's value: that is a
# sweep setting, not a contract, so a test that relies on it silently flips
# meaning when the config is re-pointed.
_OFF = ["model.forward_model.inlet_turbulence.enabled=false", _SEED_ON]


def _make_model(tmp_path: pathlib.Path, *extra_overrides: str) -> Any:
    """Compose + instantiate a pypalm smoke model staged under ``tmp_path``."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="run_forward_model",
            overrides=[*_SMOKE, f"paths.experiment_dir={tmp_path}", *extra_overrides],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        return instantiate(cfg.model.forward_model)


def _static_params(angle: float = 30.0, speed: float = 5.0) -> xarray.Dataset:
    return xarray.Dataset(
        data_vars={"inflow_angle": float(angle), "velocity_magnitude": float(speed)}
    )


def _time_varying_params(times: Any, angles: Any, speeds: Any = 5.0) -> xarray.Dataset:
    def _var(v: Any) -> Any:
        return ("time", np.asarray(v, dtype=float)) if np.ndim(v) else float(v)

    return xarray.Dataset(
        data_vars={"inflow_angle": _var(angles), "velocity_magnitude": _var(speeds)},
        coords={"time": np.asarray(times, dtype=float)},
    )


def _get(fm: Any, section: str, key: str) -> Optional[str]:
    from pypalm.utils.p3d_utils import P3DFile

    return P3DFile(fm.p3d_path).get_value(section, key)


def _rt(fm: Any, key: str) -> Optional[str]:
    return _get(fm, "runtime_parameters", key)


def _init(fm: Any, key: str) -> Optional[str]:
    return _get(fm, "initialization_parameters", key)


def _rt_float(fm: Any, key: str) -> float:
    raw = _rt(fm, key)
    assert raw is not None, f"runtime_parameters/{key} not written"
    return float(raw)


def _init_int(fm: Any, key: str) -> int:
    raw = _init(fm, key)
    assert raw is not None, f"initialization_parameters/{key} not written"
    return int(raw)


def _turbulent_inflow_active(fm: Any) -> bool:
    from pypalm.utils.p3d_utils import P3DFile

    p3d = P3DFile(fm.p3d_path)
    if not p3d.has_section("turbulent_inflow_parameters"):
        return False
    return p3d.get_value("turbulent_inflow_parameters", "switch_off_module") != ".true."


# The keys the knob owns; used to assert the disabled path is a strict no-op.
_DISTURBANCE_KEYS = (
    ("runtime_parameters", "dt_disturb"),
    ("runtime_parameters", "disturbance_amplitude"),
    ("runtime_parameters", "disturbance_energy_limit"),
    ("runtime_parameters", "disturbance_level_b"),
    ("runtime_parameters", "disturbance_level_t"),
    ("initialization_parameters", "inflow_disturbance_begin"),
    ("initialization_parameters", "inflow_disturbance_end"),
)


# --- absent / disabled is a strict no-op ------------------------------------


def test_default_config_writes_no_disturbance_keys(tmp_path: pathlib.Path) -> None:
    """`enabled: false` writes only create_disturbances (per initial_seed)."""
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW, *_OFF)
    assert not fm.inlet_turbulence_enabled

    fm._apply_inflow_settings(_static_params())
    fm._reset_cold_init()
    before = fm.p3d_path.read_text()
    fm._apply_inlet_turbulence(warm_start=False)

    assert fm.p3d_path.read_text() == before
    for section, key in _DISTURBANCE_KEYS:
        assert _get(fm, section, key) is None, key


def test_absent_config_is_a_no_op(tmp_path: pathlib.Path) -> None:
    """`inlet_turbulence=None` behaves exactly like `enabled: false`."""
    fm = _make_model(
        tmp_path, *_INFLOW_OUTFLOW, "~model.forward_model.inlet_turbulence"
    )
    assert fm._inlet_turbulence is None
    assert not fm.inlet_turbulence_enabled

    fm._apply_inflow_settings(_static_params())
    before = fm.p3d_path.read_text()
    assert fm._apply_inlet_turbulence(warm_start=False) is False
    assert fm.p3d_path.read_text() == before


def test_disabled_with_seed_true_reclaims_create_disturbances(
    tmp_path: pathlib.Path,
) -> None:
    """`initial_seed` now owns create_disturbances, so a stale .false. is fixed.

    Previously this key was left to the warm/cold init path, which is why
    `enabled: false` still fired PALM's t=0 kick. Now the knob is authoritative:
    a `.false.` left by an earlier `initial_seed: false` run in the same
    experiment dir must not leak into this one.
    """
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW, *_OFF)
    fm._apply_inflow_settings(_static_params())

    fm._p3d_set_value("runtime_parameters", "create_disturbances", False)
    fm._apply_inlet_turbulence(warm_start=False)
    assert _rt(fm, "create_disturbances") == ".true."


def test_disabled_with_seed_false_turns_it_fully_off(
    tmp_path: pathlib.Path,
) -> None:
    """`enabled: false` + `initial_seed: false` = no disturbances at all."""
    fm = _make_model(
        tmp_path,
        *_INFLOW_OUTFLOW,
        *_OFF[:1],
        _SEED_OFF,
    )
    fm._apply_inflow_settings(_static_params())
    fm._reset_cold_init()  # writes create_disturbances=.true.
    fm._apply_inlet_turbulence(warm_start=False)
    assert _rt(fm, "create_disturbances") == ".false."


# --- enabled: the knob writes PALM's inflow-disturbance settings -------------


def test_enabled_writes_disturbance_namelist(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW, *_ON)
    assert fm.inlet_turbulence_enabled

    fm._apply_inflow_settings(_static_params())
    assert fm._apply_inlet_turbulence(warm_start=False) is True

    assert _rt(fm, "create_disturbances") == ".true."
    assert _rt_float(fm, "dt_disturb") == pytest.approx(5.0)
    assert _rt_float(fm, "disturbance_amplitude") == pytest.approx(0.25)
    # Cold start keeps PALM's default energy limit -> the initial kick still
    # fires, exactly as today.
    assert _rt_float(fm, "disturbance_energy_limit") == pytest.approx(0.01)
    # Unset tunables are left to PALM's own auto-derivation.
    assert _init(fm, "inflow_disturbance_begin") is None
    assert _init(fm, "inflow_disturbance_end") is None


def test_enabled_tunables_are_written(tmp_path: pathlib.Path) -> None:
    fm = _make_model(
        tmp_path,
        *_INFLOW_OUTFLOW,
        *_ON,
        "model.forward_model.inlet_turbulence.dt_disturb=2.5",
        "model.forward_model.inlet_turbulence.amplitude=0.4",
        "model.forward_model.inlet_turbulence.begin=3",
        "model.forward_model.inlet_turbulence.end=12",
        "model.forward_model.inlet_turbulence.level_b=1.5",
        "model.forward_model.inlet_turbulence.level_t=6.0",
    )
    fm._apply_inflow_settings(_static_params())
    fm._apply_inlet_turbulence(warm_start=False)

    assert _rt_float(fm, "dt_disturb") == pytest.approx(2.5)
    assert _rt_float(fm, "disturbance_amplitude") == pytest.approx(0.4)
    assert _init_int(fm, "inflow_disturbance_begin") == 3
    assert _init_int(fm, "inflow_disturbance_end") == 12
    assert _rt_float(fm, "disturbance_level_b") == pytest.approx(1.5)
    assert _rt_float(fm, "disturbance_level_t") == pytest.approx(6.0)


def test_enabled_warm_start_suppresses_initial_kick(tmp_path: pathlib.Path) -> None:
    """Warm start: energy limit 0.0 kills the init kick, keeps the in-run loop.

    init_3d_model.f90:1488 gates the one-off perturbation on
    ``create_disturbances .AND. disturbance_energy_limit /= 0.0``; the in-run
    branch at time_integration.f90:976 fires precisely when the limit is 0.0.
    """
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW, *_ON)
    fm._apply_inflow_settings(_static_params())
    # Mirror _apply_warmstart, which disables disturbances outright.
    fm._p3d_set_value("runtime_parameters", "create_disturbances", False)
    fm._apply_inlet_turbulence(warm_start=True)

    assert _rt(fm, "create_disturbances") == ".true."
    assert _rt_float(fm, "disturbance_energy_limit") == 0.0
    assert _rt_float(fm, "dt_disturb") == pytest.approx(5.0)


def test_enabled_then_disabled_restores_palm_defaults(tmp_path: pathlib.Path) -> None:
    """A prior enabled run in the same experiment dir cannot leak its settings."""
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW, *_ON)
    fm._apply_inflow_settings(_static_params())
    fm._apply_inlet_turbulence(warm_start=False)

    fm._inlet_turbulence = {"enabled": False}
    assert fm._apply_inlet_turbulence(warm_start=False) is False

    from pypalm.utils import inlet_turbulence_utils as itu

    assert _rt_float(fm, "dt_disturb") == pytest.approx(itu.PALM_DEFAULT_DT_DISTURB)
    assert _rt_float(fm, "disturbance_energy_limit") == pytest.approx(
        itu.PALM_DEFAULT_ENERGY_LIMIT
    )


# --- composition with the existing inflow drivers ---------------------------


def test_enabled_does_not_disturb_the_dynamic_driver(tmp_path: pathlib.Path) -> None:
    """Inlet turbulence composes with time-varying inflow, and leaves it alone.

    PALM's ``turbulent_inflow`` (method ``read_from_file``) is the *reader* for
    the ``_dynamic`` driver carrying inflow_angle/velocity_magnitude. The knob
    must never interfere with it in either direction.
    """
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW, *_ON)
    fm._apply_inflow_settings(
        _time_varying_params([0.0, 50.0, 100.0], [0.0, 30.0, 60.0])
    )
    fm._apply_inlet_turbulence(warm_start=False)

    assert fm.dynamic_driver_path.exists()
    assert _turbulent_inflow_active(fm)
    assert _rt(fm, "create_disturbances") == ".true."


def test_disabled_never_switches_off_turbulent_inflow(tmp_path: pathlib.Path) -> None:
    """`enabled: false` must NOT sever the time-varying inflow signal.

    Rejected-by-design combination: there is no "turbulent_inflow off + keep the
    time-varying inflow" mode, because that module *is* the driver reader.
    """
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)  # knob off (shipped default)
    fm._apply_inflow_settings(
        _time_varying_params([0.0, 50.0, 100.0], [0.0, 30.0, 60.0])
    )
    fm._apply_inlet_turbulence(warm_start=False)

    assert _turbulent_inflow_active(fm)
    assert fm.dynamic_driver_path.exists()


# --- guards -----------------------------------------------------------------


def test_periodic_plus_enabled_raises(tmp_path: pathlib.Path) -> None:
    """PALM keeps no inflow strip under cyclic BCs — reject rather than no-op."""
    import shutil

    from pypalm.forward_model import ForwardModel

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    shutil.copy2(pathlib.Path("examples/palm/xie_and_castro/_p3d"), case_dir / "_p3d")

    with pytest.raises(ValueError, match="inflow_outflow"):
        ForwardModel(
            case_dir=case_dir,
            stl_path="examples/xie_and_castro/xie_castro_2008_STL.stl",
            boundary_condition="periodic",
            inlet_turbulence={"enabled": True},
            temp_dir=tmp_path / "exp",
            verbose=False,
        )


def test_non_positive_dt_disturb_raises(tmp_path: pathlib.Path) -> None:
    fm = _make_model(
        tmp_path,
        *_INFLOW_OUTFLOW,
        *_ON,
        "model.forward_model.inlet_turbulence.dt_disturb=0.0",
    )
    fm._apply_inflow_settings(_static_params())
    with pytest.raises(ValueError, match="dt_disturb"):
        fm._apply_inlet_turbulence(warm_start=False)


# ---------------------------------------------------------------------------
# initial_seed is independent of enabled: the cold-start init path writes
# create_disturbances=.true. unconditionally, so `enabled: false` alone used to
# leave PALM's t=0 random kick firing (init_3d_model.f90:1488).
# ---------------------------------------------------------------------------


def _apply(tmp_path: pathlib.Path, block, warm_start: bool = False) -> dict:
    """Stage a template, mimic the cold-start init write, apply the knob."""
    import shutil

    from pypalm.utils.inlet_turbulence_utils import apply_inlet_turbulence
    from pypalm.utils.p3d_utils import P3DFile

    p = tmp_path / "urban_run_p3d"
    shutil.copy2(pathlib.Path("examples/palm/xie_and_castro/_p3d"), p)
    # _reset_cold_init() does this on EVERY cold start, before the knob runs.
    f = P3DFile(p)
    f.set_value("runtime_parameters", "create_disturbances", True)
    f.write()

    apply_inlet_turbulence(p, block, warm_start=warm_start)
    f = P3DFile(p)
    return {
        k: f.get_value("runtime_parameters", k)
        for k in ("create_disturbances", "dt_disturb", "disturbance_energy_limit")
    }


def _kick_fires(v: dict) -> bool:
    """init_3d_model.f90:1488 — create_disturbances AND energy_limit /= 0."""
    if v["create_disturbances"] != ".true.":
        return False
    lim = v["disturbance_energy_limit"]
    return lim is None or float(lim) != 0.0


def _in_run_fires(v: dict) -> bool:
    """time_integration.f90:966 — create_disturbances AND dt_disturb reachable."""
    if v["create_disturbances"] != ".true.":
        return False
    dt = v["dt_disturb"]
    return dt is not None and float(dt) < 1.0e6


@pytest.mark.parametrize(
    "seed,enabled,want_kick,want_in_run",
    [
        (True, False, True, False),  # historical default: seed only
        (True, True, True, True),  # both
        (False, False, False, False),  # genuinely off
        (False, True, False, True),  # continuous forcing, no initial shock
    ],
)
def test_seed_and_enabled_are_independent(
    tmp_path: pathlib.Path,
    seed: bool,
    enabled: bool,
    want_kick: bool,
    want_in_run: bool,
) -> None:
    vals = _apply(tmp_path, {"enabled": enabled, "initial_seed": seed})
    assert _kick_fires(vals) is want_kick
    assert _in_run_fires(vals) is want_in_run


def test_absent_config_keeps_the_historical_seed(tmp_path: pathlib.Path) -> None:
    """A config that never mentions inlet_turbulence must not change physics."""
    vals = _apply(tmp_path, None)
    assert _kick_fires(vals) is True
    assert _in_run_fires(vals) is False


def test_warm_start_suppresses_the_kick_even_with_seed_true(
    tmp_path: pathlib.Path,
) -> None:
    """Warm starts must not shock the injected field, whatever initial_seed says."""
    vals = _apply(tmp_path, {"enabled": True, "initial_seed": True}, warm_start=True)
    assert _kick_fires(vals) is False
    assert _in_run_fires(vals) is True
