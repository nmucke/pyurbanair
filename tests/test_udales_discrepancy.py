"""Fast contract and algebra checks for the native Vreman discrepancy."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import xarray as xr
from pyudales.utils.discrepancy_utils import (
    DISCREPANCY_NAMELIST_KEYS,
    DISCREPANCY_PARAM_NAMES,
    extract_discrepancy_coefficients,
    height_feature,
    strain_rotation_feature,
    validate_model_discrepancy,
    viscosity_multiplier,
    write_model_discrepancy,
)
from pyudales.utils.namoptions_utils import NamoptionsFile

ENABLED = {
    "enabled": True,
    "canopy_height": 20.0,
    "height_band_over_H": [0.5, 1.5],
    "gradient_regularization": 0.01,
    "log_multiplier_cap": float(np.log(3)),
}
VREMAN = "&NAMSUBGRID\nlsmagorinsky = .false.\nlvreman = .true.\nc_vreman = 0.07\n/\n"


@pytest.mark.parametrize("config", [None, {}, {"enabled": False}])  # type: ignore[misc]
def test_disabled_namelist_is_byte_identical(
    tmp_path: pathlib.Path, config: dict | None
) -> None:
    path = tmp_path / "namoptions.999"
    original = "! untouched formatting\n&NAMSUBGRID\n  cs=0.15 ! comment\n/\n"
    path.write_text(original)
    assert write_model_discrepancy(path, config) is None
    assert path.read_text() == original


@pytest.mark.parametrize(  # type: ignore[misc]
    "override, match",
    [
        ({"canopy_height": None}, "canopy_height"),
        ({"canopy_height": 0}, "canopy_height"),
        ({"gradient_regularization": float("nan")}, "gradient_regularization"),
        ({"gradient_regularization": -1}, "gradient_regularization"),
        ({"log_multiplier_cap": 1000}, "native REAL"),
        ({"height_band_over_H": [1, 1]}, "z_b > z_a"),
        ({"height_band_over_H": [1, float("inf")]}, "height_band_over_H"),
        ({"height_band_over_H": [1]}, "two finite heights"),
        ({"canopy_height": 1e308, "height_band_over_H": [1, 10]}, "physical width"),
        ({"kind": "acceleration"}, "kind"),
        ({"coefficient_model": "ou"}, "persistent"),
        ({"prior_std": [1, 1, 0]}, "prior_std"),
        ({"prior_std": [1, 1]}, "three positive"),
        ({"b3": 0}, "Unknown"),
        ({"enabled": "false"}, "boolean"),
    ],
)
def test_invalid_settings(override: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_model_discrepancy({**ENABLED, **override})


def test_fixed_run_needs_no_prior() -> None:
    settings = validate_model_discrepancy(ENABLED)
    assert settings["coefficient_model"] == "persistent"
    assert "prior_std" not in settings
    assert validate_model_discrepancy({**ENABLED, "prior_std": [0.1, 0.2, 0.3]})[
        "prior_std"
    ] == [0.1, 0.2, 0.3]


def test_scalar_extractor_is_separate_from_inflow() -> None:
    from pyudales.utils.params_utils import extract_inflow_params

    params = xr.Dataset({"sgs_bias_b1": 0.3, "inflow_angle": 90.0})
    assert extract_discrepancy_coefficients(params) == {
        "sgs_bias_b0": 0.0,
        "sgs_bias_b1": 0.3,
        "sgs_bias_b2": 0.0,
    }
    inflow = extract_inflow_params(params)
    assert inflow is not None
    assert list(inflow.data_vars) == ["inflow_angle"]
    assert extract_discrepancy_coefficients(None) == dict.fromkeys(
        DISCREPANCY_PARAM_NAMES, 0.0
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])  # type: ignore[misc]
def test_extractor_rejects_invalid_scalars(value: float) -> None:
    with pytest.raises(ValueError, match="sgs_bias_b0"):
        extract_discrepancy_coefficients(xr.Dataset({"sgs_bias_b0": value}))


@pytest.mark.parametrize("dim", ["time", "ensemble", "x"])  # type: ignore[misc]
def test_extractor_rejects_arrays_even_length_one(dim: str) -> None:
    with pytest.raises(ValueError, match="constant during a forecast"):
        extract_discrepancy_coefficients(xr.Dataset({"sgs_bias_b0": (dim, [0.1])}))


def test_writes_all_coefficients_and_resets_missing_values(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "namoptions.999"
    path.write_text(VREMAN)
    metadata = write_model_discrepancy(path, ENABLED, xr.Dataset({"sgs_bias_b0": 0.8}))
    assert metadata is not None
    assert metadata["coefficients"]["sgs_bias_b0"] == 0.8
    assert metadata["height_band_m"] == [10.0, 30.0]
    assert metadata["multiplier_bounds"] == pytest.approx([1 / 3, 3])
    json.dumps(metadata, allow_nan=False)
    first = NamoptionsFile(path)
    assert first.get_value_as_bool("NAMSUBGRID", "lsgs_discrepancy")
    assert first.get_value_as_float("NAMSUBGRID", "sgs_bias_b0") == 0.8
    write_model_discrepancy(path, ENABLED)
    reset = NamoptionsFile(path)
    for name in DISCREPANCY_PARAM_NAMES:
        assert reset.get_value_as_float("NAMSUBGRID", name) == 0.0
    assert reset.get_value_as_float("NAMSUBGRID", "c_vreman") == 0.07
    write_model_discrepancy(path, None)
    disabled = NamoptionsFile(path)
    assert not set(DISCREPANCY_NAMELIST_KEYS) & set(
        disabled.get_section_keys("NAMSUBGRID")
    )
    assert disabled.get_value_as_float("NAMSUBGRID", "c_vreman") == 0.07
    before = path.read_bytes()
    write_model_discrepancy(path, None)
    assert path.read_bytes() == before


def test_removes_case_insensitive_stale_keys(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "namoptions.999"
    path.write_text("&namsubgrid\nLSGS_DISCREPANCY = .true.\nSGS_BIAS_B0 = 0.3\n/\n")
    write_model_discrepancy(path, None)
    assert "DISCREPANCY" not in path.read_text()
    assert "SGS_BIAS" not in path.read_text()


@pytest.mark.parametrize(  # type: ignore[misc]
    "switches",
    [
        "lsmagorinsky = .true.\n",
        "lvreman = .false.\n",
        "lsmagorinsky = .false.\nlvreman = .false.\n",
        "lvreman = .true.\nloneeqn = .true.\n",
        "lsmagorinsky = .true.\nlvreman = .true.\n",
    ],
)
def test_requires_active_vreman(tmp_path: pathlib.Path, switches: str) -> None:
    path = tmp_path / "namoptions.999"
    original = f"&NAMSUBGRID\n{switches}/\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="active Vreman"):
        write_model_discrepancy(path, ENABLED)
    assert path.read_text() == original


def test_native_closure_defaults_are_vreman(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "namoptions.999"
    path.write_text("&NAMSUBGRID\n/\n")
    assert write_model_discrepancy(path, ENABLED) is not None


def test_namelist_removal_can_be_undone(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "namoptions.999"
    path.write_text(VREMAN)
    namoptions = NamoptionsFile(path)
    assert not namoptions.remove_value("NAMSUBGRID", "missing")
    assert namoptions.remove_value("NAMSUBGRID", "c_vreman")
    namoptions.set_value("NAMSUBGRID", "c_vreman", 0.1)
    namoptions.write()
    assert NamoptionsFile(path).get_value_as_float("NAMSUBGRID", "c_vreman") == 0.1


def test_invariant_known_gradients() -> None:
    zero = np.zeros((3, 3))
    strain = np.diag([2.0, -2.0, 0.0])
    rotation = np.array([[0, -2, 0], [2, 0, 0], [0, 0, 0]])
    shear = np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]])
    gradients = np.stack([zero, strain, rotation, shear])
    q = strain_rotation_feature(gradients, 0.1)
    np.testing.assert_allclose(q, [0, -8 / 8.01, 8 / 8.01, 0])
    np.testing.assert_array_equal(
        q, strain_rotation_feature(gradients.swapaxes(-1, -2), 0.1)
    )
    assert np.all(np.abs(q) < 1)
    # Gradient and regularizer both have inverse-time units.
    np.testing.assert_allclose(q, strain_rotation_feature(gradients * 100, 10))
    assert np.isfinite(strain_rotation_feature(strain * 1e300, 0.1))


def test_height_support_and_scale() -> None:
    z = np.array([-1, 10, 15, 20, 25, 30, 100])
    np.testing.assert_allclose(
        height_feature(z, 20, [0.5, 1.5]), [0, 0, 0.5, 1, 0.5, 0, 0]
    )
    np.testing.assert_allclose(
        height_feature(z * 2, 40, [0.5, 1.5]), height_feature(z, 20, [0.5, 1.5])
    )
    assert height_feature(20, 20, [0.5, 1.5]) == 1


def test_multiplier_bounds_zero_and_linear_limit() -> None:
    q = np.linspace(-1, 1, 100)
    phi = np.linspace(0, 1, 100)
    cap = float(np.log(3))
    np.testing.assert_array_equal(
        viscosity_multiplier(q, phi, [0, 0, 0], cap), np.ones(100)
    )
    actual = viscosity_multiplier(q, phi, [0, 1000, -2000], cap)
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= np.exp(-cap))
    assert np.all(actual <= np.exp(cap))
    np.testing.assert_array_equal(0 * actual, np.zeros(100))
    coeffs = [1e-6, 2e-6, -3e-6]
    np.testing.assert_allclose(
        viscosity_multiplier(q, phi, coeffs, cap),
        1 + coeffs[0] + coeffs[1] * phi + coeffs[2] * q,
        atol=2e-11,
        rtol=0,
    )


@pytest.mark.parametrize("gradient", [np.ones((2, 2)), np.full((3, 3), np.inf)])  # type: ignore[misc]
def test_gradient_shape_and_finiteness(gradient: np.ndarray) -> None:
    with pytest.raises(ValueError, match="gradient"):
        strain_rotation_feature(gradient, 0.1)
