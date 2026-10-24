"""The ``roughness_length`` knob: wall z0 written into uDALES' factypes.inp.

The unit tests stage a real experiment dir (no preprocessing, no solver) and
add the factypes.inp the python preprocessing writes, so the format is the one
uDALES reads. The integration test runs the tiny case at two z0 values. See
docs/pyudales.md "Roughness".
"""

import logging
import os
import pathlib
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import xarray
from pyudales.forward_model import ForwardModel
from pyudales.python_udgeom.preprocessing import Preprocessing

CASE = pathlib.Path(__file__).resolve().parents[2] / "geometries" / "xie_and_castro"
PARAMS = {"inflow_angle": 0.0, "velocity_magnitude": 3.0}


def _write_inputs(experiment_dir: pathlib.Path, expnr: str, ktot: int) -> None:
    """The preprocessing outputs the staging edits: factypes.inp as write_inputs
    writes it, every facet of type 1, and empty prof/lscale profiles."""
    obj = SimpleNamespace(nfaclyrs=3, expnr=expnr)
    obj.addvar = lambda name, value: setattr(obj, name, value)
    Preprocessing.generate_factypes(obj)
    cwd = pathlib.Path.cwd()
    try:
        os.chdir(experiment_dir)
        Preprocessing.write_factypes(obj)
    finally:
        os.chdir(cwd)
    (experiment_dir / f"facets.inp.{expnr}").write_text(
        "# type, normal\n" + "1    0.0000 0.0000 1.0000\n" * 4
    )
    heights = (np.arange(ktot) + 0.5) * 2.0
    (experiment_dir / f"prof.inp.{expnr}").write_text(
        "# test\n# z thl qt u v tke\n"
        + "".join(f"  {z:20.15f}  288.0  0.0  0.0  0.0  0.0\n" for z in heights)
    )
    (experiment_dir / f"lscale.inp.{expnr}").write_text(
        "# test\n# z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad\n"
        + "".join(f"  {z:20.15f}" + "  0.0" * 9 + "\n" for z in heights)
    )


def _model(tmp_path: pathlib.Path, **kwargs: object) -> ForwardModel:
    case = tmp_path / "case"
    shutil.copytree(CASE, case)
    model = ForwardModel(
        case_dir=case,
        temp_dir=tmp_path / "temp",
        ncpu=1,
        nx=20,
        ny=20,
        nz=8,
        bounds=((0.0, 40.0), (0.0, 40.0), (0.0, 16.0)),
        boundary_condition="inflow_outflow",
        verbose=False,
        **kwargs,  # type: ignore[arg-type]
    )
    _write_inputs(model.dirs.experiment_dir, model.dirs.experiment_name, ktot=8)
    return model


def _factypes(model: ForwardModel) -> pathlib.Path:
    return model.dirs.experiment_dir / f"factypes.inp.{model.dirs.experiment_name}"


def _z0(model: ForwardModel) -> dict[int, float]:
    """z0 per type id, read the way uDALES reads it (3 header lines)."""
    table = np.loadtxt(_factypes(model), skiprows=3)
    return {int(row[0]): float(row[2]) for row in table}


def test_default_is_byte_identical(tmp_path: pathlib.Path) -> None:
    """No knob, or the default 0.05 the file already holds: nothing written."""
    for name, kwargs in (("absent", {}), ("default", {"roughness_length": 0.05})):
        model = _model(tmp_path / name, **kwargs)
        before = _factypes(model).read_bytes()
        model._apply_roughness(xarray.Dataset(PARAMS))
        assert _factypes(model).read_bytes() == before
        assert "  0.05  0.00035" in before.decode()


def test_params_value_is_written_to_types_in_use(tmp_path: pathlib.Path) -> None:
    model = _model(tmp_path, roughness_length=0.05)
    before = np.loadtxt(_factypes(model), skiprows=3)
    model._apply_roughness(xarray.Dataset({**PARAMS, "roughness_length": 0.0123}))

    after = np.loadtxt(_factypes(model), skiprows=3)
    z0 = _z0(model)
    # Type 1 is the only one in use; the free-slip bounding walls (-101) and
    # the unused types keep theirs, and every other column is unchanged.
    assert z0[1] == pytest.approx(0.0123, abs=1e-9)
    assert z0[-101] == 0.0 and z0[-1] == 0.05 and z0[2] == 0.05
    changed = before != after
    assert changed.sum() == 1 and changed[:, 2].sum() == 1
    assert "  0.012300  0.00035" in _factypes(model).read_text()


def test_model_config_value_applies_without_params(tmp_path: pathlib.Path) -> None:
    model = _model(tmp_path, roughness_length=0.002)
    model._apply_roughness(xarray.Dataset(PARAMS))
    # Below the preprocessing's %6.2f resolution, which would round it to 0.
    assert _z0(model)[1] == pytest.approx(0.002)


def test_ensemble_members_get_their_own_value(tmp_path: pathlib.Path) -> None:
    from pyudales.utils.forward_model_utils import create_new_forward_model

    base = _model(tmp_path, roughness_length=0.05)
    base_bytes = _factypes(base).read_bytes()
    members = [
        create_new_forward_model(base, tmp_path / "ensemble", f"{i:03d}")
        for i in range(2)
    ]
    for member, z0 in zip(members, (0.02, 0.15)):
        member._apply_inflow_settings(
            xarray.Dataset({**PARAMS, "roughness_length": z0})
        )
    assert _z0(members[0])[1] == pytest.approx(0.02)
    assert _z0(members[1])[1] == pytest.approx(0.15)
    assert _factypes(base).read_bytes() == base_bytes


def test_warm_start_reapplies_after_a_window_restore(tmp_path: pathlib.Path) -> None:
    """``run_single`` calls ``_apply_inflow_settings`` every window; the member's
    merged params keep z0, so a factypes.inp put back (e.g. by the window
    checkpoint restore) gets it again."""
    model = _model(tmp_path)
    pristine = _factypes(model).read_bytes()
    model._apply_inflow_settings(xarray.Dataset({**PARAMS, "roughness_length": 0.08}))
    assert _z0(model)[1] == pytest.approx(0.08)

    _factypes(model).write_bytes(pristine)
    model._apply_inflow_settings(None, warm_start=True)
    assert _z0(model)[1] == pytest.approx(0.08)


@pytest.mark.parametrize("value", [0.0, -0.01, 4e-7, np.nan, np.inf])  # type: ignore[misc]
def test_invalid_values_raise(tmp_path: pathlib.Path, value: float) -> None:
    model = _model(tmp_path)
    before = _factypes(model).read_bytes()
    with pytest.raises(ValueError, match="roughness_length"):
        model._apply_roughness(xarray.Dataset({**PARAMS, "roughness_length": value}))
    assert _factypes(model).read_bytes() == before


def test_invalid_model_config_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="roughness_length"):
        _model(tmp_path, roughness_length=0.0)


def test_missing_factypes_raises(tmp_path: pathlib.Path) -> None:
    model = _model(tmp_path)
    _factypes(model).unlink()
    with pytest.raises(FileNotFoundError, match="factypes"):
        model._apply_roughness(xarray.Dataset({**PARAMS, "roughness_length": 0.1}))


def test_warns_when_most_sections_are_within_e_z0(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = _model(tmp_path)
    (model.dirs.experiment_dir / "facet_sections_u.txt").write_text(
        "# facet, area, fluid boundary point, distance\n"
        + "1 0.5 1 0.10\n" * 3
        + "1 0.5 2 0.80\n"
    )
    with caplog.at_level(logging.WARNING, logger="pyudales.forward_model"):
        model._apply_roughness(xarray.Dataset({**PARAMS, "roughness_length": 0.02}))
    # e*z0 = 0.054 m: no section that close.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    with caplog.at_level(logging.WARNING, logger="pyudales.forward_model"):
        model._apply_roughness(xarray.Dataset({**PARAMS, "roughness_length": 0.1}))
    assert "75% of the IBM wall sections" in caplog.text


def test_wall_and_ground_split_retypes_walls(tmp_path: pathlib.Path) -> None:
    """Walls of a type shared with horizontal facets move to a new type with the
    wall z0; ground and roofs keep their type with the ground z0. Idempotent."""
    model = _model(tmp_path)
    facets = model.dirs.experiment_dir / f"facets.inp.{model.dirs.experiment_name}"
    facets.write_text(
        "# type, normal\n"
        "1    0.0000 0.0000 1.0000\n"  # ground
        "1    1.0000 0.0000 0.0000\n"  # wall
        "1    0.0000 -1.0000 0.0000\n"  # wall
        "-101 1.0000 0.0000 0.0000\n"  # free-slip bounding wall
    )
    before = np.loadtxt(_factypes(model), skiprows=3)
    split = {"roughness_length_wall": 0.2, "roughness_length_ground": 0.01}
    model._apply_roughness(xarray.Dataset({**PARAMS, **split}))

    new = int(before[:, 0].max()) + 1
    types = np.loadtxt(facets, skiprows=1, usecols=0).astype(int)
    assert types.tolist() == [1, new, new, -101]
    z0 = _z0(model)
    assert z0[1] == pytest.approx(0.01) and z0[new] == pytest.approx(0.2)
    assert z0[-101] == 0.0
    after = np.loadtxt(_factypes(model), skiprows=3)
    original = after[after[:, 0] == 1][0]
    wall_row = after[after[:, 0] == new][0]
    # The new type copies type 1's row apart from its id and z0.
    assert np.array_equal(np.delete(wall_row, [0, 2]), np.delete(original, [0, 2]))

    files = (_factypes(model).read_bytes(), facets.read_bytes())
    model._apply_roughness(xarray.Dataset({**PARAMS, **split}))
    assert (_factypes(model).read_bytes(), facets.read_bytes()) == files


@pytest.mark.parametrize(  # type: ignore[misc]
    "params",
    [
        {"roughness_length_wall": 0.1},
        {"roughness_length_ground": 0.1},
        {
            "roughness_length": 0.1,
            "roughness_length_wall": 0.1,
            "roughness_length_ground": 0.1,
        },
    ],
)
def test_partial_or_mixed_split_raises(
    tmp_path: pathlib.Path, params: dict[str, float]
) -> None:
    model = _model(tmp_path)
    with pytest.raises(ValueError, match="roughness_length_wall"):
        model._apply_roughness(xarray.Dataset({**PARAMS, **params}))


@pytest.mark.integration  # type: ignore[misc]
def test_e2e_rougher_walls_slow_the_near_wall_flow(tmp_path: pathlib.Path) -> None:
    """The tiny case at two z0: the higher one gives the lower near-wall speed."""
    from hydra.utils import instantiate

    from tests.conftest import compose

    speeds = {}
    for z0 in (0.01, 0.1):
        cfg = compose(
            "forward",
            "+test=forward",
            "model=pyudales_stock",
            f"flow.roughness_length={z0}",
            "time.simulation_time=6.0",
            "time.spinup_time=6.0",
            root=tmp_path / str(z0),
        )
        fm = instantiate(cfg.model.forward_model)
        instantiate(cfg.model.prepare, forward_model=fm)
        state = fm.run_single(params=xarray.Dataset(PARAMS))
        assert _z0(fm)[1] == pytest.approx(z0)
        # Streamwise wind in the first cell above the ground (inflow along
        # x), averaged over the window and the plane.
        u = state["u"].isel(zt=0).mean("time").values
        assert np.isfinite(u).all()
        speeds[z0] = float(u.mean())
    print(f"near-wall mean u by z0: {speeds}")
    assert speeds[0.1] < speeds[0.01]
