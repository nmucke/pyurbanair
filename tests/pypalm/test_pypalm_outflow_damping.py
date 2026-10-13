"""Staging tests for pypalm's ``outflow_damping`` knob.

Like ``test_palm_inlet_turbulence.py`` these compose a real smoke model and
check the ``_p3d`` namelist WITHOUT running PALM. inflow_outflow runs get
PALM's outflow damping layer (``outflow_damping_factor`` /
``outflow_damping_width`` in ``&initialization_parameters``); periodic runs must
never carry a positive factor (PALM PAC0352). See docs/pypalm.md §8.
"""

import pathlib
import shutil
from typing import Any

import pytest
from hydra.errors import InstantiationException
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
    # Pin the values the tests assert on rather than inheriting the config's.
    "model.forward_model.outflow_damping.factor=1.0",
    "model.forward_model.outflow_damping.width=4.0",
]

# inflow_outflow selects the multigrid solver: ncpu must divide nx.
_INFLOW_OUTFLOW = [
    "model.forward_model.boundary_condition=inflow_outflow",
    "model.forward_model.ncpu=4",
]
_PERIODIC = ["model.forward_model.boundary_condition=periodic"]

_SECTION = "initialization_parameters"
_KEYS = ("outflow_damping_factor", "outflow_damping_width")


def _make_model(tmp_path: pathlib.Path, *extra_overrides: str) -> Any:
    cfg = compose("forward", "+test=forward", *_SMOKE, *extra_overrides, root=tmp_path)
    return instantiate(cfg.model.forward_model)


def _p3d(fm: Any) -> Any:
    from pypalm.utils.p3d_utils import P3DFile

    return P3DFile(fm.p3d_path)


def _count(fm: Any, key: str) -> int:
    return sum(
        1
        for line in fm.p3d_path.read_text().splitlines()
        if line.strip().split("=")[0].strip() == key
    )


def _write_stale_keys(fm: Any) -> None:
    """Simulate a template / reused _p3d that already carries the keys."""
    p3d = _p3d(fm)
    p3d.set_value(_SECTION, "outflow_damping_factor", 0.5)
    p3d.set_value(_SECTION, "outflow_damping_width", 3.0)
    p3d.write()


def test_inflow_outflow_writes_both_keys_once(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    p3d = _p3d(fm)
    assert float(p3d.get_value(_SECTION, "outflow_damping_factor")) == 1.0
    assert float(p3d.get_value(_SECTION, "outflow_damping_width")) == 4.0
    for key in _KEYS:
        assert _count(fm, key) == 1, key
        for other in p3d.sections:
            if other != _SECTION:
                assert key not in p3d.sections[other], (other, key)


def test_inflow_outflow_overwrites_template_keys(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_INFLOW_OUTFLOW)
    _write_stale_keys(fm)
    fm._apply_boundary_condition()
    p3d = _p3d(fm)
    assert float(p3d.get_value(_SECTION, "outflow_damping_factor")) == 1.0
    assert float(p3d.get_value(_SECTION, "outflow_damping_width")) == 4.0
    for key in _KEYS:
        assert _count(fm, key) == 1, key


def test_periodic_writes_neither_key(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, *_PERIODIC)
    for key in _KEYS:
        assert _p3d(fm).get_value(_SECTION, key) is None, key


def test_periodic_is_byte_identical_without_the_block(tmp_path: pathlib.Path) -> None:
    with_block = _make_model(tmp_path / "a", *_PERIODIC)
    without = _make_model(
        tmp_path / "b", *_PERIODIC, "model.forward_model.outflow_damping=null"
    )
    assert with_block.p3d_path.read_text() == without.p3d_path.read_text()


def test_periodic_zeroes_stale_keys(tmp_path: pathlib.Path) -> None:
    """A positive factor under cyclic BCs is PALM error PAC0352."""
    fm = _make_model(tmp_path, *_PERIODIC)
    _write_stale_keys(fm)
    fm._apply_boundary_condition()
    for key in _KEYS:
        assert float(_p3d(fm).get_value(_SECTION, key)) == 0.0, key
        assert _count(fm, key) == 1, key


def test_null_opts_out(tmp_path: pathlib.Path) -> None:
    fm = _make_model(
        tmp_path, *_INFLOW_OUTFLOW, "model.forward_model.outflow_damping=null"
    )
    for key in _KEYS:
        assert _p3d(fm).get_value(_SECTION, key) is None, key


@pytest.mark.parametrize(
    "override, match",
    [
        ("model.forward_model.outflow_damping.factor=-0.1", "factor"),
        ("model.forward_model.outflow_damping.width=0.0", "width"),
        ("model.forward_model.outflow_damping.width=25.0", "domain length"),
    ],
)
def test_invalid_values_raise(
    tmp_path: pathlib.Path, override: str, match: str
) -> None:
    # Hydra wraps the constructor's ValueError.
    with pytest.raises(InstantiationException, match=match):
        _make_model(tmp_path, *_INFLOW_OUTFLOW, override)


def test_width_under_two_cells_warns(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The smoke grid has dx = 1 m.
    with caplog.at_level("WARNING", logger="pypalm.forward_model"):
        _make_model(
            tmp_path, *_INFLOW_OUTFLOW, "model.forward_model.outflow_damping.width=1.5"
        )
    assert "under two cells" in caplog.text


def test_direct_construction_without_the_kwarg_writes_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """Absent (the constructor default) is a strict no-op."""
    from pypalm.forward_model import ForwardModel

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    shutil.copy2(pathlib.Path("geometries/xie_and_castro/_p3d"), case_dir / "_p3d")
    fm = ForwardModel(
        case_dir=case_dir,
        stl_path="geometries/xie_and_castro/xie_castro_2008_STL.stl",
        boundary_condition="inflow_outflow",
        temp_dir=tmp_path / "exp",
        verbose=False,
    )
    for key in _KEYS:
        assert _p3d(fm).get_value(_SECTION, key) is None, key
