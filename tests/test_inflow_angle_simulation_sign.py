"""Simulation-based sign checks for LBM inflow angle conventions.

These tests use actual forward-model runs to compare the direct LBM Fortran
input path against the Python wrapper, using inlet-plane mean v as the most
direct observable of the imposed inflow direction.
"""

import pytest
import xarray
from hydra.utils import instantiate
from pylbm.forward_model import ForwardModel
from pylbm.utils.infile_utils import Infile
from pyurbanair.config.hydra_helpers import clean_outputs


def _get_v_mean(state: xarray.Dataset, x_index: int | None = None) -> float:
    """Return the latest-timestep mean v, optionally at a single x-plane."""
    if "time" in state.dims:
        state = state.isel(time=-1)

    v_name = "v" if "v" in state.data_vars else "v0"

    if v_name not in state.data_vars:
        raise AssertionError(
            f"Could not find velocity components in state. data_vars={list(state.data_vars)}"
        )

    v = state[v_name]
    if x_index is not None:
        v = v.isel(x=x_index)

    return float(v.mean())


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
def pylbm_cfg(compose_module_cfg):
    """Compose a single-model pylbm test config once for this module."""
    return compose_module_cfg(
        [
            "model=pylbm",
            "model.forward_model.cuda=false",
        ]
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def pylbm_model(pylbm_cfg) -> ForwardModel:
    """Create and compile pylbm once for this module."""
    model = instantiate(pylbm_cfg.model.forward_model)
    instantiate(pylbm_cfg.model.prepare, forward_model=model)
    return model  # type: ignore[no-any-return]


@pytest.fixture(scope="module")  # type: ignore[misc]
def true_velocity_magnitude(pylbm_cfg) -> float:
    return float(pylbm_cfg.params.true.velocity_magnitude)


@pytest.mark.parametrize("angle_deg,expected_v_sign", [(20.0, -1.0), (-20.0, 1.0)])  # type: ignore[misc]
def test_lbm_fortran_inverts_input_udir_sign(
    pylbm_model: ForwardModel,
    true_velocity_magnitude: float,
    angle_deg: float,
    expected_v_sign: float,
) -> None:
    """Direct LBM Fortran input produces the opposite inlet-v sign."""
    clean_outputs(model_name="pylbm", forward_model=pylbm_model)
    _set_lbm_inflow_direct(
        model=pylbm_model,
        velocity_magnitude=true_velocity_magnitude,
        angle_deg=angle_deg,
    )

    state = pylbm_model.run_single(params=None)
    v_mean = _get_v_mean(state, x_index=0)

    assert expected_v_sign * v_mean > 0.0, (
        f"Direct LBM udir={angle_deg} produced inlet mean v={v_mean:.6f}; "
        "expected the opposite sign based on the observed solver convention."
    )


@pytest.mark.parametrize("angle_deg", [20.0, -20.0])  # type: ignore[misc]
def test_lbm_wrapper_corrects_direct_fortran_sign_convention(
    pylbm_model: ForwardModel,
    true_velocity_magnitude: float,
    angle_deg: float,
) -> None:
    """Wrapper path should correct the sign inversion seen in direct LBM input.

    A (wrapper): run_single(params=...) writes a corrected udir for LBM.
    B (direct):  write udir=angle directly in infile and run run_single(params=None).
    """
    velocity = true_velocity_magnitude

    # A) Wrapper path (current production behavior)
    clean_outputs(model_name="pylbm", forward_model=pylbm_model)
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": angle_deg,
            "velocity_magnitude": velocity,
        }
    )
    state_wrapper = pylbm_model.run_single(params=params)
    v_mean_wrapper = _get_v_mean(state_wrapper, x_index=0)

    # B) Direct Fortran input path
    clean_outputs(model_name="pylbm", forward_model=pylbm_model)
    _set_lbm_inflow_direct(
        model=pylbm_model,
        velocity_magnitude=velocity,
        angle_deg=angle_deg,
    )
    state_direct = pylbm_model.run_single(params=None)
    v_mean_direct = _get_v_mean(state_direct, x_index=0)

    sign_wrapper = _sign(v_mean_wrapper)
    sign_direct = _sign(v_mean_direct)
    sign_target = _sign(angle_deg)

    assert sign_direct == -sign_target, (
        f"Direct LBM run should invert input sign. angle={angle_deg}, "
        f"v_mean_direct={v_mean_direct:.6f}"
    )
    assert sign_wrapper == sign_target, (
        f"Wrapper run should restore the user-facing sign convention. "
        f"angle={angle_deg}, v_mean_wrapper={v_mean_wrapper:.6f}"
    )
    assert sign_wrapper == -sign_direct, (
        f"A/B regression failed: wrapper should flip the direct LBM sign. "
        f"v_mean_wrapper={v_mean_wrapper:.6f}, v_mean_direct={v_mean_direct:.6f}"
    )
