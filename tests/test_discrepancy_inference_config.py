"""Static SGS discrepancy inference configuration, without a CFD solve."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.discrepancy import (
    SGS_BIAS_PARAMETER_METADATA,
    SGS_BIAS_PARAMETER_NAMES,
    augment_sgs_discrepancy_prior,
    validate_sgs_discrepancy_inference,
    validate_sgs_discrepancy_settings,
)


def _prior() -> DictConfig:
    return OmegaConf.create(
        {
            "_target_": "pyurbanair.static_parameters.ParameterSampler",
            "_convert_": "all",
            "seed": 123,
            "parameters": {
                "inflow_angle": {
                    "_target_": "pyurbanair.static_parameters.Normal",
                    "mean": 0.0,
                    "std": 1.0,
                },
                "velocity_magnitude": {
                    "_target_": "pyurbanair.static_parameters.Constant",
                    "value": 5.0,
                },
            },
        }
    )


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


def _run_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "assim_model": {
                "name": "pyudales",
                "forward_model": {
                    "closure": "vreman",
                    "model_discrepancy": OmegaConf.to_container(_discrepancy()),
                },
            },
            "prior_params": OmegaConf.to_container(_prior()),
            "esmda": {
                "num_assimilation_windows": 2,
                "smoother": {
                    "_target_": "data_assimilation.smoothing.esmda.ParameterESMDA"
                },
            },
            "filtering": {"mode": "joint", "parameter_evolution": None},
            "ensemble": {"failure": {"policy": "raise"}},
            "params_to_estimate": list(SGS_BIAS_PARAMETER_NAMES),
        }
    )


def test_augmentation_preserves_existing_draws_and_truth_schema() -> None:
    prior = _prior()
    truth = OmegaConf.create({"parameters": {"velocity_magnitude": {"value": 5.0}}})
    original_prior = copy.deepcopy(prior)
    augmented = augment_sgs_discrepancy_prior(prior, _discrepancy())

    assert prior == original_prior
    assert list(augmented.parameters) == [
        "inflow_angle",
        "velocity_magnitude",
        *SGS_BIAS_PARAMETER_NAMES,
    ]
    assert list(truth.parameters) == ["velocity_magnitude"]
    for name, std in zip(SGS_BIAS_PARAMETER_NAMES, (0.1, 0.2, 0.3)):
        assert OmegaConf.to_container(augmented.parameters[name]) == {
            "_target_": "pyurbanair.static_parameters.Normal",
            "mean": 0.0,
            "std": std,
        }
        assert SGS_BIAS_PARAMETER_METADATA[name] == {
            "scope": "global",
            "localization": "global",
        }

    baseline_draw = instantiate(prior).sample(2048)
    augmented_draw = instantiate(augmented).sample(2048)
    for name in prior.parameters:
        np.testing.assert_array_equal(baseline_draw[name], augmented_draw[name])
    coefficients = np.column_stack(
        [augmented_draw[name].values for name in SGS_BIAS_PARAMETER_NAMES]
    )
    np.testing.assert_allclose(coefficients.mean(axis=0), 0, atol=0.02)
    np.testing.assert_allclose(coefficients.std(axis=0), [0.1, 0.2, 0.3], rtol=0.08)
    assert np.max(np.abs(np.corrcoef(coefficients, rowvar=False) - np.eye(3))) < 0.08


@pytest.mark.parametrize(  # type: ignore[misc]
    "disabled", [None, {}, {"enabled": False}]
)
def test_disabled_augmentation_returns_original_config(disabled: Any) -> None:
    prior = _prior()
    assert augment_sgs_discrepancy_prior(prior, disabled) is prior


@pytest.mark.parametrize(  # type: ignore[misc]
    "scales",
    [
        None,
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
        augment_sgs_discrepancy_prior(_prior(), discrepancy)
    cfg = _run_cfg()
    cfg.assim_model.forward_model.model_discrepancy.prior_std = scales
    with pytest.raises(ValueError, match="prior_std"):
        validate_sgs_discrepancy_inference(cfg, "esmda")


def test_existing_coefficient_distribution_and_dynamic_prior_rejected() -> None:
    prior = _prior()
    prior.parameters.sgs_bias_b1 = {
        "_target_": "pyurbanair.static_parameters.Constant",
        "value": 0.0,
    }
    with pytest.raises(ValueError, match="already defines.*sgs_bias_b1"):
        augment_sgs_discrepancy_prior(prior, _discrepancy())
    prior._target_ = "pyurbanair.dynamic_parameters.ar2_relaxation.AR2RelaxationModel"
    with pytest.raises(ValueError, match="static ParameterSampler"):
        augment_sgs_discrepancy_prior(prior, _discrepancy())
    cfg = _run_cfg()
    cfg.prior_params.parameters.sgs_bias_b1 = {"_target_": "prior", "value": 0.0}
    with pytest.raises(ValueError, match="already defines.*sgs_bias_b1"):
        validate_sgs_discrepancy_inference(cfg, "esmda")


def test_supported_scope_allows_multiple_windows_and_prior_only_coefficients() -> None:
    cfg = _run_cfg()
    validate_sgs_discrepancy_inference(cfg, "esmda")
    assert cfg.esmda.num_assimilation_windows == 2
    assert "sgs_bias_b0" not in cfg.prior_params.parameters


@pytest.mark.parametrize(  # type: ignore[misc]
    "path,value,error",
    [
        ("assim_model.name", "pypalm", "pyudales backend"),
        ("assim_model.forward_model.closure", "smagorinsky", "Vreman closure"),
        ("assim_model.forward_model.model_discrepancy.kind", "other", "kind"),
        (
            "assim_model.forward_model.model_discrepancy.coefficient_model",
            "dynamic",
            "persistent",
        ),
        ("prior_params._target_", "other", "static prior sampler"),
        (
            "esmda.smoother._target_",
            "data_assimilation.smoothing.esmda.StateESMDA",
            "parameter-only",
        ),
        ("params_to_estimate", ["sgs_bias_b0", "sgs_bias_b1"], "exactly"),
        ("params_to_estimate", [*SGS_BIAS_PARAMETER_NAMES, "sgs_constant"], "exactly"),
        ("params_to_estimate", None, "exactly"),
    ],
)
def test_unsupported_inference_scope_rejected(
    path: str, value: Any, error: str
) -> None:
    cfg = _run_cfg()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError, match=error):
        validate_sgs_discrepancy_inference(cfg, "esmda")


@pytest.mark.parametrize("mode", ["parameter", "joint"])  # type: ignore[misc]
def test_filtering_scope_accepts_parameter_updates_with_identity_evolution(
    mode: str,
) -> None:
    cfg = _run_cfg()
    cfg.filtering.mode = mode
    validate_sgs_discrepancy_inference(cfg, "filtering")
    cfg.filtering.parameter_evolution = {
        "_target_": "data_assimilation.filtering.parameter_evolution.IdentityEvolution"
    }
    validate_sgs_discrepancy_inference(cfg, "filtering")


def test_filtering_rejects_state_only_or_coefficient_random_walk() -> None:
    cfg = _run_cfg()
    cfg.filtering.mode = "state"
    with pytest.raises(ValueError, match="parameter or joint"):
        validate_sgs_discrepancy_inference(cfg, "filtering")
    cfg.filtering.mode = "joint"
    cfg.filtering.parameter_evolution = {
        "_target_": "data_assimilation.filtering.parameter_evolution.RandomWalkEvolution",
        "std": 0.1,
    }
    with pytest.raises(ValueError, match="identity parameter evolution"):
        validate_sgs_discrepancy_inference(cfg, "filtering")


def test_parameter_only_filter_rejects_distance_localization() -> None:
    cfg = _run_cfg()
    cfg.filtering.mode = "parameter"
    cfg.filtering.localization = {
        "_target_": "data_assimilation.localization.distance.DistanceLocalization"
    }
    with pytest.raises(ValueError, match="cannot use distance localization"):
        validate_sgs_discrepancy_inference(cfg, "filtering")


def test_hybrid_requires_state_only_filter_after_static_parameter_esmda() -> None:
    cfg = _run_cfg()
    cfg.filtering.mode = "state"
    validate_sgs_discrepancy_inference(cfg, "filter_smoothing")
    cfg.filtering.mode = "joint"
    with pytest.raises(ValueError, match="filtering.mode=state"):
        validate_sgs_discrepancy_inference(cfg, "filter_smoothing")
    cfg.filtering.mode = "state"
    cfg.ensemble.failure.policy = "resample_from_successes"
    with pytest.raises(ValueError, match="failure.policy=raise"):
        validate_sgs_discrepancy_inference(cfg, "filter_smoothing")


def test_other_workflow_rejected() -> None:
    with pytest.raises(ValueError, match="only ESMDA, filtering"):
        validate_sgs_discrepancy_inference(_run_cfg(), "run_forward_model")


def test_disabled_inference_has_no_mode_restrictions() -> None:
    cfg = _run_cfg()
    cfg.assim_model.forward_model.model_discrepancy.enabled = False
    cfg.esmda.smoother._target_ = "anything"
    validate_sgs_discrepancy_inference(cfg, "filtering")


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
    cfg = _run_cfg()
    cfg.assim_model.forward_model.model_discrepancy = discrepancy
    cfg.assim_model.forward_model.model_discrepancy.prior_std = [0.1, 0.2, 0.3]
    validate_sgs_discrepancy_inference(cfg, "esmda")
    discrepancy.log_multiplier_cap = MAX_LOG_MULTIPLIER_CAP + 1.0e-4
    with pytest.raises(ValueError, match="log_multiplier_cap"):
        validate_sgs_discrepancy_settings(discrepancy)
