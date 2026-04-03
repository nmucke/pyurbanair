"""Parity tests for inflow-angle conventions across pyudales and pylbm.

These tests do not run CFD models. They validate the parameter-convention
mapping used by the Python wrappers:
  - pyudales interprets inflow_angle directly via cos/sin(angle)
  - pylbm negates inflow_angle before writing infile udir
"""

import pathlib
from types import SimpleNamespace

import xarray
from pylbm.utils.infile_utils import Infile
from pylbm.utils.params_utils import apply_inflow_settings as apply_lbm_inflow_settings

_INFILE_TEMPLATE = """# minimal infile for test
 6.0 0.0          ! uini, udir      : Inflow wind velocity [m/s], direction in degrees
"""


def _make_test_infile(tmp_path: pathlib.Path) -> pathlib.Path:
    infile_path = tmp_path / "infile.in"
    infile_path.write_text(_INFILE_TEMPLATE)
    return infile_path


def _read_udir_from_infile(infile_path: pathlib.Path) -> float:
    infile = Infile(infile_path)
    key = next(k for k in infile.get_keys() if k.startswith("uini"))
    value = infile.get_value(key)
    if value is None:
        raise AssertionError("Could not read uini/udir from test infile.")
    uini_token, udir_token = value.split()[:2]
    _ = float(uini_token)  # sanity parse
    return float(udir_token)


def test_pylbm_negates_inflow_angle_before_writing(tmp_path: pathlib.Path) -> None:
    """pylbm wrapper should negate inflow_angle before writing infile udir."""
    infile_path = _make_test_infile(tmp_path)
    dirs = SimpleNamespace(infile_path=infile_path)

    params = xarray.Dataset(
        data_vars={
            "inflow_angle": 30.0,
            "velocity_magnitude": 5.0,
        }
    )
    apply_lbm_inflow_settings(params=params, dirs=dirs)  # type: ignore[arg-type]

    udir_written = _read_udir_from_infile(infile_path)
    assert udir_written == -30.0
