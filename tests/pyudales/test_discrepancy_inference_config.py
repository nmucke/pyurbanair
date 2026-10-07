"""SGS discrepancy settings validation, without a CFD solve."""

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.discrepancy import validate_sgs_discrepancy_settings


def _discrepancy() -> DictConfig:
    return OmegaConf.create(
        {
            "enabled": True,
            "kind": "sgs_strain_rotation",
            "coefficient_model": "persistent",
            "canopy_height": 10.0,
            "height_band_over_H": [0.5, 2.0],
            "gradient_regularization": 0.01,
            "log_multiplier_cap": 0.5,
            "prior_std": [0.1, 0.2, 0.3],
        }
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "scales",
    [
        [],
        [0.1, 0.2],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0, 0.3],
        [0.1, -0.2, 0.3],
        [0.1, float("nan"), 0.3],
        [0.1, float("inf"), 0.3],
        [0.1, "0.2", 0.3],
        [0.1, True, 0.3],
    ],
)
def test_invalid_prior_scales_rejected(scales: Any) -> None:
    discrepancy = _discrepancy()
    discrepancy.prior_std = scales
    with pytest.raises(ValueError, match="prior_std"):
        validate_sgs_discrepancy_settings(discrepancy)


@pytest.mark.parametrize(  # type: ignore[misc]
    "path,value,error",
    [
        ("canopy_height", None, "canopy_height"),
        ("canopy_height", 0.0, "canopy_height"),
        ("canopy_height", float("inf"), "canopy_height"),
        ("gradient_regularization", -1.0, "gradient_regularization"),
        ("gradient_regularization", float("nan"), "gradient_regularization"),
        ("log_multiplier_cap", 0, "log_multiplier_cap"),
        ("log_multiplier_cap", 100.0, "log_multiplier_cap"),
        ("height_band_over_H", None, "height_band_over_H"),
        ("height_band_over_H", [0.5], "height_band_over_H"),
        ("height_band_over_H", [2.0, 0.5], "z_b > z_a"),
        ("height_band_over_H", [0.5, float("nan")], "height_band_over_H"),
        ("height_band_over_H", [0.0, float("inf")], "height_band_over_H"),
        ("height_band_over_H", [1.0e308, 1.1e308], "physical width"),
    ],
)
def test_feature_settings_rejected_at_preflight(
    path: str, value: Any, error: str
) -> None:
    discrepancy = _discrepancy()
    OmegaConf.update(discrepancy, path, value)
    with pytest.raises(ValueError, match=error):
        validate_sgs_discrepancy_settings(discrepancy)


def test_feature_preflight_matches_native_cap_and_optional_defaults() -> None:
    from pyudales.utils.discrepancy_utils import (
        MAX_LOG_MULTIPLIER_CAP,
        validate_model_discrepancy,
    )

    discrepancy = _discrepancy()
    del discrepancy.kind
    del discrepancy.coefficient_model
    discrepancy.prior_std = None
    discrepancy.log_multiplier_cap = MAX_LOG_MULTIPLIER_CAP
    validate_sgs_discrepancy_settings(discrepancy)
    validate_model_discrepancy(OmegaConf.to_container(discrepancy))
    discrepancy.log_multiplier_cap = MAX_LOG_MULTIPLIER_CAP + 1.0e-4
    with pytest.raises(ValueError, match="log_multiplier_cap"):
        validate_sgs_discrepancy_settings(discrepancy)
