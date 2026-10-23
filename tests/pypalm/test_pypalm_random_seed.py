"""Staging tests for pypalm's optional ``random_seed`` knob.

Like ``test_palm_inlet_turbulence.py`` these compose a real smoke model but do
not run PALM: ``random_seed`` only writes ``ensemble_member_nr`` into
``&initialization_parameters`` of the staged ``_p3d``. Why that key is PALM's
seed is documented in docs/pypalm.md §7.
"""

import pathlib
from typing import Any

import pytest
import xarray
from hydra.utils import instantiate

from tests.conftest import compose

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
    "model.forward_model.inlet_turbulence.enabled=true",
    "model.forward_model.inlet_turbulence.initial_seed=true",
]


def _make_model(
    tmp_path: pathlib.Path, *extra_overrides: str, drop_seed: bool = False
) -> Any:
    cfg = compose("forward", "+test=forward", *_SMOKE, *extra_overrides, root=tmp_path)
    if drop_seed:
        # Hydra's `~key` override can't delete a null value, so drop it here:
        # the model is then built exactly as before the knob existed.
        del cfg.model.forward_model.random_seed
    return instantiate(cfg.model.forward_model)


def _stage_cold_run(fm: Any) -> str:
    """Stage everything ``run_single`` writes before launching PALM."""
    params = xarray.Dataset(data_vars={"inflow_angle": 30.0, "velocity_magnitude": 5.0})
    fm._apply_inflow_settings(params)
    fm._reset_cold_init()
    fm._apply_inlet_turbulence(warm_start=False)
    return str(fm.p3d_path.read_text())


def test_null_seed_is_byte_identical_to_absent(tmp_path: pathlib.Path) -> None:
    absent = _make_model(tmp_path / "absent", drop_seed=True)
    null = _make_model(tmp_path / "null", "model.forward_model.random_seed=null")
    assert absent.random_seed is None and null.random_seed is None

    text = _stage_cold_run(null)
    assert text == _stage_cold_run(absent)
    assert "ensemble_member_nr" not in text


def test_seed_is_written_once_to_initialization_parameters(
    tmp_path: pathlib.Path,
) -> None:
    from pypalm.utils.p3d_utils import P3DFile

    fm = _make_model(tmp_path, "model.forward_model.random_seed=7")
    text = _stage_cold_run(fm)

    assert text.count("ensemble_member_nr") == 1
    p3d = P3DFile(fm.p3d_path)
    assert p3d.get_value("initialization_parameters", "ensemble_member_nr") == "7"
    assert p3d.get_value("runtime_parameters", "ensemble_member_nr") is None


def test_ensemble_members_inherit_the_seed(tmp_path: pathlib.Path) -> None:
    from pypalm.utils.forward_model_utils import create_new_forward_model
    from pypalm.utils.p3d_utils import P3DFile

    fm = _make_model(tmp_path, "model.forward_model.random_seed=7")
    member = create_new_forward_model(fm, tmp_path / "ensemble", "member_0")
    value = P3DFile(member.p3d_path).get_value(
        "initialization_parameters", "ensemble_member_nr"
    )
    assert value == "7"


@pytest.mark.parametrize(  # type: ignore[misc]
    ("value", "error"),
    [
        (-1, ValueError),
        (2001, ValueError),
        (1.5, TypeError),
        ("7", TypeError),
        (True, TypeError),
    ],
)
def test_invalid_seed_raises(value: Any, error: type[Exception]) -> None:
    from pypalm.forward_model import validate_random_seed

    with pytest.raises(error, match="random_seed"):
        validate_random_seed(value)


def test_invalid_seed_fails_at_construction(tmp_path: pathlib.Path) -> None:
    with pytest.raises(Exception, match="random_seed"):
        _make_model(tmp_path, "model.forward_model.random_seed=-1")


def test_seed_rejects_a_generator_that_ignores_it(tmp_path: pathlib.Path) -> None:
    """'numerical-recipes' draws from a fixed iran, so the seed would be lost."""
    fm = _make_model(tmp_path, "model.forward_model.random_seed=7")
    fm._p3d_set_string(
        "initialization_parameters", "random_generator", "numerical-recipes"
    )
    with pytest.raises(ValueError, match="random-parallel"):
        fm._apply_random_seed()


def test_seed_without_any_disturbance_raises(tmp_path: pathlib.Path) -> None:
    """With no disturbance at all PALM never draws a random number: periodic
    runs (no inflow perturbations) with initial_seed off are bit-identical for
    every seed, so a set seed is rejected rather than silently ignored."""
    with pytest.raises(Exception, match="initial_seed"):
        _make_model(
            tmp_path,
            "model.forward_model.boundary_condition=periodic",
            "model.forward_model.inlet_turbulence.enabled=false",
            "model.forward_model.inlet_turbulence.initial_seed=false",
            "model.forward_model.random_seed=7",
        )


def test_periodic_seed_with_initial_kick_is_written(tmp_path: pathlib.Path) -> None:
    from pypalm.utils.p3d_utils import P3DFile

    fm = _make_model(
        tmp_path,
        "model.forward_model.boundary_condition=periodic",
        "model.forward_model.inlet_turbulence.enabled=false",
        "model.forward_model.random_seed=7",
    )
    p3d = P3DFile(fm.p3d_path)
    assert p3d.get_value("initialization_parameters", "ensemble_member_nr") == "7"
