"""The ground boundary knob: ``ground_boundary`` -> the ``kbnd`` line of infile.in.

``kbnd`` picks the closed-boundary treatment of the k = 1 (first digit) and
k = nz (second digit) planes in ``m_boundarycond.F90``: 1 is half-way
bounce-back (no-slip), 2 specular reflection (free-slip).
"""

import pathlib
import re
from typing import Any, Optional

import pytest
import xarray
from pylbm.utils.infile_utils import Infile
from pylbm.utils.warm_start_utils import identify_latest_restart_iteration

REPO = pathlib.Path(__file__).resolve().parents[2]
LBM_SRC = REPO / "libs" / "pylbm" / "LBM" / "src"
STL_PATH = REPO / "geometries" / "xie_and_castro" / "xie_castro_2008_STL.stl"

pytestmark = pytest.mark.skipif(
    not (LBM_SRC / "m_mkinfile.F90").exists(), reason="LBM sources not checked out"
)


def _solver_template_infile() -> str:
    """The infile.in the solver writes for itself (m_mkinfile.F90)."""
    source = (LBM_SRC / "m_mkinfile.F90").read_text()
    return "\n".join(re.findall(r"write\(10,'\(a\)'\)'(.*)'", source)) + "\n"


def _staged_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> Any:
    """compile(compile=False) on the solver's own template, without a binary."""
    from pylbm.forward_model import ForwardModel

    model = ForwardModel(
        stl_path=STL_PATH,
        temp_dir=tmp_path,
        nx=12,
        ny=10,
        nz=4,
        bounds=((0.0, 12.0), (0.0, 10.0), (0.0, 4.0)),
        cuda=False,
        verbose=False,
        boundary_condition="inflow_outflow",
        **kwargs,
    )
    model.dirs.infile_path.write_text(_solver_template_infile())
    monkeypatch.setattr(model, "_verify_prebuilt_binary", lambda: None)
    model.compile(compile=False)
    return model


def _staged_infile(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> pathlib.Path:
    return pathlib.Path(_staged_model(tmp_path, monkeypatch, **kwargs).dirs.infile_path)


@pytest.mark.parametrize(  # type: ignore[misc]
    "ground, kbnd", [("no_slip", 12), ("free_slip", 22)]
)
def test_ground_boundary_sets_kbnd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, ground: str, kbnd: int
) -> None:
    path = _staged_infile(tmp_path, monkeypatch, ground_boundary=ground)
    assert Infile(path).get_value_as_int("kbnd") == kbnd


def test_absent_leaves_the_solver_template(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None is a no-op: the template's kbnd (22) and every other line stay."""
    path = _staged_infile(tmp_path, monkeypatch)
    assert Infile(path).get_value_as_int("kbnd") == 22


def test_only_the_kbnd_line_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """infile.in is read positionally, so the knob must touch one line only."""
    base = _staged_infile(tmp_path / "a", monkeypatch).read_text().splitlines()
    wall = (
        _staged_infile(tmp_path / "b", monkeypatch, ground_boundary="no_slip")
        .read_text()
        .splitlines()
    )
    assert len(base) == len(wall)
    changed = [(old, new) for old, new in zip(base, wall) if old != new]
    assert len(changed) == 1
    assert "kbnd" in changed[0][1] and changed[0][1].split()[0] == "12"


@pytest.mark.parametrize("bad", ["noslip", "rough", ""])  # type: ignore[misc]
def test_unknown_value_raises(tmp_path: pathlib.Path, bad: Optional[str]) -> None:
    from pylbm.forward_model import ForwardModel

    with pytest.raises(ValueError, match="ground_boundary"):
        ForwardModel(
            stl_path=STL_PATH,
            temp_dir=tmp_path,
            nx=12,
            ny=10,
            nz=4,
            bounds=((0.0, 12.0), (0.0, 10.0), (0.0, 4.0)),
            cuda=False,
            ground_boundary=bad,
        )


class _Launched(Exception):
    """Raised instead of running the solver, once everything is staged."""


@pytest.mark.parametrize(  # type: ignore[misc]
    "ground, from_restart", [("no_slip", True), ("free_slip", False)]
)
def test_a_no_slip_cold_start_is_a_restart_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    ground: str,
    from_restart: bool,
) -> None:
    """Only a restart sets the ground's ghost corners, so a laminar no-slip
    cold start is staged as one (the state the solver's cold start builds)."""
    model = _staged_model(
        tmp_path,
        monkeypatch,
        ground_boundary=ground,
        profile_config={"type": "power_law", "alpha": 0.25, "z_ref": 8.0},
        simulation_time=2.0,
        output_frequency=1.0,
        spinup_time=1.0,
    )

    def launched() -> None:
        raise _Launched

    monkeypatch.setattr(model, "run", launched)
    params = xarray.Dataset({"inflow_angle": 10.0, "velocity_magnitude": 6.0})
    with pytest.raises(_Launched):
        model.run_single(params=params)

    nt0 = Infile(model.dirs.infile_path).get_value_as_int("nt0")
    if from_restart:
        assert nt0 == model._window_timesteps > 0
        assert identify_latest_restart_iteration(model.dirs) == nt0
    else:
        assert nt0 == 0
        assert identify_latest_restart_iteration(model.dirs) is None
