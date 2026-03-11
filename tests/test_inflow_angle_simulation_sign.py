"""Simulation-based sign checks for inflow angle conventions.

These tests use actual forward-model runs to determine how each solver responds
to positive/negative input angles by measuring the sign of domain-mean v.
"""

import pathlib
import sys

import pytest
import xarray
from pylbm.forward_model import ForwardModel
from pylbm.utils.infile_utils import Infile

# Patch scripts.config to use tests.config before any script imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config

from pyurbanair.utils.config_utils import (
    clean_forward_model_outputs,
    create_forward_model,
    prepare_forward_model,
)


def _get_uv_means(state: xarray.Dataset) -> tuple[float, float]:
    """Return domain-mean (u, v) from the latest timestep."""
    if "time" in state.dims:
        state = state.isel(time=-1)

    u_name = "u" if "u" in state.data_vars else "u0"
    v_name = "v" if "v" in state.data_vars else "v0"

    if u_name not in state.data_vars or v_name not in state.data_vars:
        raise AssertionError(
            f"Could not find velocity components in state. data_vars={list(state.data_vars)}"
        )

    return float(state[u_name].mean()), float(state[v_name].mean())


def _set_lbm_inflow_direct(
    model: ForwardModel, velocity_magnitude: float, angle_deg: float
) -> None:
    """Write uini/udir directly in infile.in to test LBM Fortran behavior."""
    infile = Infile(model.dirs.infile_path)
    uini_key = next((k for k in infile.get_keys() if k.startswith("uini")), None)
    if uini_key is None:
        raise AssertionError("Could not find uini/udir key in infile.in")
    infile.set_value(uini_key, f"{velocity_magnitude:.1f} {angle_deg:.1f}")
    infile.write()


def _sign(value: float) -> int:
    """Return -1, 0, or +1 depending on the sign of value."""
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


@pytest.fixture(scope="module")  # type: ignore[misc]
def pylbm_model() -> ForwardModel:
    """Create and compile pylbm once for this module."""
    model = create_forward_model(model_name="pylbm")
    prepare_forward_model(model_name="pylbm", forward_model=model)
    return model  # type: ignore[no-any-return]


@pytest.mark.parametrize("angle_deg,expected_v_sign", [(20.0, 1.0), (-20.0, -1.0)])  # type: ignore[misc]
def test_lbm_fortran_respects_input_udir_sign(
    pylbm_model: ForwardModel, angle_deg: float, expected_v_sign: float
) -> None:
    """LBM Fortran should produce v with same sign as directly supplied udir."""
    clean_forward_model_outputs(model_name="pylbm", forward_model=pylbm_model)
    _set_lbm_inflow_direct(
        model=pylbm_model,
        velocity_magnitude=tests_config.TRUE_PARAMS["velocity_magnitude"],
        angle_deg=angle_deg,
    )

    state = pylbm_model.run_single(params=None)
    _, v_mean = _get_uv_means(state)

    assert expected_v_sign * v_mean > 0.0, (
        f"LBM direct udir={angle_deg} produced mean v={v_mean:.6f}; "
        "this suggests sign inversion in simulation response."
    )


@pytest.mark.parametrize("angle_deg", [20.0, -20.0])  # type: ignore[misc]
def test_lbm_ab_wrapper_vs_direct_inflow_angle_regression(
    pylbm_model: ForwardModel, angle_deg: float
) -> None:
    """A/B regression: wrapper path should match direct-Fortran input.

    A (wrapper): run_single(params=...) uses pylbm Python mapping (angle).
    B (direct):  write udir=angle directly in infile and run run_single(params=None).
    """
    velocity = tests_config.TRUE_PARAMS["velocity_magnitude"]

    # A) Wrapper path (current production behavior)
    clean_forward_model_outputs(model_name="pylbm", forward_model=pylbm_model)
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": angle_deg,
            "velocity_magnitude": velocity,
        }
    )
    state_wrapper = pylbm_model.run_single(params=params)
    _, v_mean_wrapper = _get_uv_means(state_wrapper)

    # B) Direct Fortran input path
    clean_forward_model_outputs(model_name="pylbm", forward_model=pylbm_model)
    _set_lbm_inflow_direct(
        model=pylbm_model,
        velocity_magnitude=velocity,
        angle_deg=angle_deg,
    )
    state_direct = pylbm_model.run_single(params=None)
    _, v_mean_direct = _get_uv_means(state_direct)

    sign_wrapper = _sign(v_mean_wrapper)
    sign_direct = _sign(v_mean_direct)
    sign_target = _sign(angle_deg)

    assert sign_direct == sign_target, (
        f"Direct LBM run should follow input sign. angle={angle_deg}, "
        f"v_mean_direct={v_mean_direct:.6f}"
    )
    assert sign_wrapper == sign_target, (
        f"Wrapper run should follow input sign. "
        f"angle={angle_deg}, v_mean_wrapper={v_mean_wrapper:.6f}"
    )
    assert sign_wrapper == sign_direct, (
        f"A/B regression failed: wrapper and direct should have the same sign. "
        f"v_mean_wrapper={v_mean_wrapper:.6f}, v_mean_direct={v_mean_direct:.6f}"
    )
