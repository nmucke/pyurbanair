"""Staging tests for pypalm's periodic nudging driver (driver-table rows).

These exercise ``ForwardModel._apply_inflow_settings`` on a real, Hydra-composed
smoke model but WITHOUT running PALM — the wiring under test is pure staging:
which files land in ``INPUT/`` and which ``_p3d`` switches flip per
(boundary_condition × params × nudging_config.enabled). See
docs/plans/palm_nudging_driver_plan.md §Phases 4.
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
# slab subdomains — ncpu must divide the grid's x point count. conf's production
# default (8) does not divide the smoke grid, so pin one that does.
_INFLOW_OUTFLOW = [
    "model.forward_model.boundary_condition=inflow_outflow",
    "model.forward_model.ncpu=4",
]

# The periodic cases must pin the BC too rather than inherit conf's default:
# conf/model/pypalm.yaml's `boundary_condition` tracks whatever sweep is being
# run, so relying on it silently re-points these tests at the other branch (a
# periodic test that actually exercises inflow_outflow still "passes" the parts
# that don't assert on the nudging apparatus).
_PERIODIC = [
    "model.forward_model.boundary_condition=periodic",
]

BOUNDS = ((0.0, 20.0), (0.0, 20.0), (0.0, 10.0))
NX = NY = 20
NZ = 4


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


def _switch(fm: Any, key: str) -> Optional[str]:
    from pypalm.utils.p3d_utils import P3DFile

    return P3DFile(fm.p3d_path).get_value("initialization_parameters", key)


def _nudging_switches_on(fm: Any) -> bool:
    return all(
        _switch(fm, k) == ".true."
        for k in ("nudging", "large_scale_forcing", "lsf_exception", "humidity")
    )


def _nudging_switches_absent_or_off(fm: Any) -> bool:
    return all(
        _switch(fm, k) in (None, ".false.")
        for k in ("nudging", "large_scale_forcing", "lsf_exception", "humidity")
    )


def _turbulent_inflow_active(fm: Any) -> bool:
    from pypalm.utils.p3d_utils import P3DFile

    p3d = P3DFile(fm.p3d_path)
    if not p3d.has_section("turbulent_inflow_parameters"):
        return False
    return p3d.get_value("turbulent_inflow_parameters", "switch_off_module") != ".true."


def _nudge_block_count(fm: Any) -> int:
    return sum(
        1 for ln in fm.nudge_driver_path.read_text().splitlines() if ln.startswith("#")
    )


# --- periodic + nudging (driver-table row 1) --------------------------------


def test_periodic_static_stages_nudging_driver(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_PERIODIC)
    fm._apply_inflow_settings(_static_params())

    assert fm.nudge_driver_path.exists() and fm.lsf_driver_path.exists()
    assert _nudging_switches_on(fm)
    assert not _turbulent_inflow_active(fm)
    assert not fm.dynamic_driver_path.exists()
    assert _nudge_block_count(fm) == 2  # static -> two snapshots


def test_periodic_time_varying_block_count(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_PERIODIC)  # spinup_time=3.0 in the smoke shape
    times = [0.0, 50.0, 100.0]
    fm._apply_inflow_settings(_time_varying_params(times, [0.0, 30.0, 60.0]))

    assert fm.nudge_driver_path.exists() and fm.lsf_driver_path.exists()
    assert _nudging_switches_on(fm)
    assert not _turbulent_inflow_active(fm)
    # one block per param time (3) + spinup plateau (1) + terminal pad (1).
    assert _nudge_block_count(fm) == len(times) + 2


# --- inflow_outflow (driver-table rows 2 & 3) -------------------------------


def test_inflow_outflow_static_no_nudging_apparatus(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    fm._apply_inflow_settings(_static_params())

    assert not fm.nudge_driver_path.exists()
    assert not fm.lsf_driver_path.exists()
    # Clean template never had the nudging switches -> staging leaves them absent
    # (byte-identical to today apart from the _nudge/_lsf cleanup).
    assert _nudging_switches_absent_or_off(fm)
    assert not fm.dynamic_driver_path.exists()


def test_inflow_outflow_time_varying_uses_dynamic_driver(
    tmp_path: pathlib.Path,
) -> None:
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    fm._apply_inflow_settings(
        _time_varying_params([0.0, 50.0, 100.0], [0.0, 30.0, 60.0])
    )

    assert fm.dynamic_driver_path.exists()  # turbulent_inflow dynamic driver
    assert _turbulent_inflow_active(fm)
    assert not fm.nudge_driver_path.exists()
    assert not fm.lsf_driver_path.exists()
    assert _nudging_switches_absent_or_off(fm)


# --- escape hatch: nudging_config.enabled=false -----------------------------


def test_enabled_false_restores_undriven_periodic_static(
    tmp_path: pathlib.Path,
) -> None:
    fm = _make_model(
        tmp_path, *_PERIODIC, "model.forward_model.nudging_config.enabled=false"
    )
    fm._apply_inflow_settings(_static_params())

    # Old un-driven static periodic path: no nudging apparatus, no dynamic driver.
    assert not fm.nudge_driver_path.exists()
    assert not fm.lsf_driver_path.exists()
    assert _nudging_switches_absent_or_off(fm)
    assert not fm.dynamic_driver_path.exists()
    assert not _turbulent_inflow_active(fm)


# --- guards -----------------------------------------------------------------


def test_periodic_nudging_requires_bounds_and_nz(tmp_path: pathlib.Path) -> None:
    """Periodic + nudging without bounds/nz raises a ValueError naming them."""
    import shutil

    from pypalm.forward_model import ForwardModel

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    shutil.copy2(pathlib.Path("examples/palm/xie_and_castro/_p3d"), case_dir / "_p3d")

    fm = ForwardModel(
        case_dir=case_dir,
        stl_path="examples/xie_and_castro/xie_castro_2008_STL.stl",
        boundary_condition="periodic",
        temp_dir=tmp_path / "exp",
        verbose=False,
    )  # nx/ny/nz/bounds all None -> topography + grid skipped
    with pytest.raises(ValueError, match="bounds and nz"):
        fm._apply_inflow_settings(_static_params())


def test_passive_scalar_with_nudging_raises(tmp_path: pathlib.Path) -> None:
    from pypalm.utils.p3d_utils import P3DFile

    fm = _make_model(tmp_path, *_PERIODIC)
    # Template enables passive_scalar -> incompatible with large_scale_forcing.
    p3d = P3DFile(fm.p3d_path)
    p3d.set_value("initialization_parameters", "passive_scalar", True)
    p3d.write()

    with pytest.raises(ValueError, match="passive_scalar"):
        fm._apply_inflow_settings(_static_params())


# --- warm-start + periodic + nudging ordering -------------------------------


def _make_state() -> xarray.Dataset:
    (xmin, _), (ymin, _), (zmin, _) = BOUNDS
    dx = (BOUNDS[0][1] - xmin) / NX
    dy = (BOUNDS[1][1] - ymin) / NY
    dz = (BOUNDS[2][1] - zmin) / NZ
    x = (np.arange(NX) + 0.5) * dx + xmin
    xu = np.arange(NX - 1) * dx + xmin
    y = (np.arange(NY) + 0.5) * dy + ymin
    yv = np.arange(NY - 1) * dy + ymin
    z = (np.arange(NZ) + 0.5) * dz + zmin

    def _arr(val: float, *lengths: int) -> np.ndarray:
        return np.full((1, *lengths), val, dtype=np.float32)

    return xarray.Dataset(
        data_vars={
            "u": (("time", "z", "y", "xu"), _arr(3.0, NZ, NY, NX - 1)),
            "v": (("time", "z", "yv", "x"), _arr(-1.0, NZ, NY - 1, NX)),
            "w": (("time", "z", "y", "x"), _arr(0.2, NZ, NY, NX)),
        },
        coords={"time": [0], "z": z, "y": y, "yv": yv, "x": x, "xu": xu},
    )


def test_warmstart_periodic_nudging_ordering(tmp_path: pathlib.Path) -> None:
    """Warm-start's LOD=2 init driver survives the nudging branch's cleanup.

    ``_apply_inflow_settings`` (which removes any stale dynamic driver) must run
    BEFORE ``_apply_warmstart`` writes the init_atmosphere_* driver — otherwise
    the warm-start field would be clobbered. Replicate run_single's ordering and
    assert the final staged state.
    """
    fm = _make_model(tmp_path, *_PERIODIC)
    # run_single disables spinup on a warm window before applying inflow settings.
    fm.disable_spinup()
    fm._apply_inflow_settings(_static_params())
    fm._apply_warmstart(_make_state())

    # Nudging apparatus staged AND the warm-start driver present + init switched.
    assert fm.nudge_driver_path.exists() and fm.lsf_driver_path.exists()
    assert _nudging_switches_on(fm)
    assert fm.dynamic_driver_path.exists()  # warm-start re-created it after cleanup
    assert _switch(fm, "initializing_actions") == "'read_from_file'"
    # Warm window sees spinup_time=0 -> static nudging still exactly two blocks.
    assert _nudge_block_count(fm) == 2
