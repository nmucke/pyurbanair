"""SGS discrepancy inference configuration, without a CFD solve."""

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


def _dynamic_prior() -> DictConfig:
    return OmegaConf.create(
        {
            "_target_": "pyurbanair.dynamic_parameters.ar2_relaxation.AR2RelaxationModel",
            "_convert_": "all",
            "seed": 123,
            "simulation_time": 10.0,
            "seconds_per_knot": 5.0,
            "correlation_length": 100.0,
            "external_parameters": {
                "inflow_angle": {
                    "_target_": "pyurbanair.static_parameters.Normal",
                    "mean": 0.0,
                    "std": 1.0,
                }
            },
            "static_parameters": {
                "velocity_magnitude": {
                    "_target_": "pyurbanair.static_parameters.Constant",
                    "value": 5.0,
                }
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


def test_explicit_coefficient_prior_takes_precedence() -> None:
    prior = _prior()
    explicit = {
        "_target_": "pyurbanair.static_parameters.Normal",
        "mean": 0.2,
        "std": 0.4,
        "min": -1.0,
    }
    prior.parameters.sgs_bias_b1 = explicit
    result = augment_sgs_discrepancy_prior(prior, _discrepancy())
    assert result.parameters.sgs_bias_b1 == explicit
    assert result.parameters.sgs_bias_b0.std == 0.1
    assert result.parameters.sgs_bias_b2.std == 0.3
    before = instantiate(prior).sample(32)
    after = instantiate(result).sample(32)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_explicit_priors_need_no_prior_std() -> None:
    cfg = _run_cfg()
    cfg.assim_model.forward_model.model_discrepancy.prior_std = None
    cfg.params_to_estimate = None
    for name in SGS_BIAS_PARAMETER_NAMES:
        cfg.prior_params.parameters[name] = {
            "_target_": "pyurbanair.static_parameters.Normal",
            "mean": 0.0,
            "std": 0.2,
            "min": -1.0,
        }
    for workflow in ["esmda", "filtering", "filter_smoothing"]:
        if workflow == "filter_smoothing":
            cfg.filtering.mode = "state"
        validate_sgs_discrepancy_inference(cfg, workflow)
    assert (
        augment_sgs_discrepancy_prior(
            cfg.prior_params, cfg.assim_model.forward_model.model_discrepancy
        )
        is cfg.prior_params
    )


def test_unselected_coefficients_are_not_added() -> None:
    prior = _prior()
    selected = ["inflow_angle", "sgs_bias_b2"]
    result = augment_sgs_discrepancy_prior(prior, _discrepancy(), selected)
    assert "sgs_bias_b0" not in result.parameters
    assert "sgs_bias_b1" not in result.parameters
    assert result.parameters.sgs_bias_b2.std == 0.3
    discrepancy = _discrepancy()
    discrepancy.prior_std = None
    assert augment_sgs_discrepancy_prior(prior, discrepancy, ["inflow_angle"]) is prior


@pytest.mark.parametrize("dynamic", [False, True])  # type: ignore[misc]
def test_configured_unestimated_coefficients_are_applied_to_independent_models(
    dynamic: bool,
) -> None:
    from pyurbanair.config.hydra_helpers import inference_parameter_configs

    cfg = _run_cfg()
    cfg.params_to_estimate = ["inflow_angle"]
    cfg.assim_model.forward_model.model_discrepancy.prior_std = None
    cfg.prior_params = _dynamic_prior() if dynamic else _prior()
    cfg.truth_params = copy.deepcopy(cfg.prior_params)
    block = "static_parameters" if dynamic else "parameters"
    for name in SGS_BIAS_PARAMETER_NAMES:
        cfg.prior_params[block][name] = {
            "_target_": "pyurbanair.static_parameters.Constant",
            "value": -20.0,
        }
        cfg.truth_params[block][name] = {
            "_target_": "pyurbanair.static_parameters.Constant",
            "value": 0.0,
        }
    original = copy.deepcopy(cfg)
    truth_cfg, prior_cfg = inference_parameter_configs(cfg)
    assert cfg == original
    truth = instantiate(truth_cfg).sample(1)
    prior = instantiate(prior_cfg).sample(4)
    assert set(truth.data_vars) == set(prior.data_vars)
    assert "velocity_magnitude" in prior
    for name in SGS_BIAS_PARAMETER_NAMES:
        np.testing.assert_array_equal(prior[name], [-20.0] * 4)
        np.testing.assert_array_equal(truth[name], [0.0])
        assert prior[name].dims == ("ensemble",)


@pytest.mark.parametrize(  # type: ignore[misc]
    "workflow", ["esmda", "filtering", "filter_smoothing"]
)
def test_prescribed_coefficients_do_not_require_coefficient_inference(
    workflow: str,
) -> None:
    cfg = _run_cfg()
    cfg.params_to_estimate = ["inflow_angle"]
    cfg.assim_model.forward_model.model_discrepancy.prior_std = None
    cfg.filtering.mode = "state"
    for name in SGS_BIAS_PARAMETER_NAMES:
        cfg.prior_params.parameters[name] = {
            "_target_": "pyurbanair.static_parameters.Constant",
            "value": -20.0,
        }
    validate_sgs_discrepancy_inference(cfg, workflow)


def test_dynamic_prior_adds_static_coefficients_without_changing_inflow() -> None:
    prior = _dynamic_prior()
    augmented = augment_sgs_discrepancy_prior(
        prior, _discrepancy(), ["inflow_angle", *SGS_BIAS_PARAMETER_NAMES]
    )
    assert list(augmented.static_parameters) == [
        "velocity_magnitude",
        *SGS_BIAS_PARAMETER_NAMES,
    ]
    assert list(prior.static_parameters) == ["velocity_magnitude"]
    before = instantiate(prior).sample(32)
    after = instantiate(augmented).sample(32)
    np.testing.assert_array_equal(before.inflow_angle, after.inflow_angle)
    assert after.inflow_angle.dims == ("time", "ensemble")
    for name in SGS_BIAS_PARAMETER_NAMES:
        assert after[name].dims == ("ensemble",)


def test_dynamic_fallback_creates_static_block_when_missing() -> None:
    prior = _dynamic_prior()
    del prior.static_parameters
    augmented = augment_sgs_discrepancy_prior(
        prior, _discrepancy(), ["inflow_angle", "sgs_bias_b0"]
    )
    assert list(augmented.static_parameters) == ["sgs_bias_b0"]
    assert instantiate(augmented).sample(4).sgs_bias_b0.dims == ("ensemble",)


def test_dynamic_explicit_coefficient_prior_needs_no_fallback_scales() -> None:
    prior = _dynamic_prior()
    prior.static_parameters.sgs_bias_b0 = {
        "_target_": "pyurbanair.static_parameters.Normal",
        "mean": 0.1,
        "std": 0.2,
    }
    discrepancy = _discrepancy()
    discrepancy.prior_std = None
    assert augment_sgs_discrepancy_prior(prior, discrepancy, ["sgs_bias_b0"]) is prior


@pytest.mark.parametrize("workflow", ["esmda", "filter_smoothing"])  # type: ignore[misc]
def test_dynamic_sgs_inference_uses_matching_parameter_smoother(workflow: str) -> None:
    cfg = _run_cfg()
    cfg.prior_params = _dynamic_prior()
    cfg.esmda.smoother._target_ = (
        "data_assimilation.smoothing.esmda.TimeVaryingParameterESMDA"
    )
    if workflow == "filter_smoothing":
        cfg.filtering.mode = "state"
    validate_sgs_discrepancy_inference(cfg, workflow)
    cfg.esmda.smoother._target_ = "data_assimilation.smoothing.esmda.ParameterESMDA"
    with pytest.raises(ValueError, match="matching the prior"):
        validate_sgs_discrepancy_inference(cfg, workflow)


def test_dynamic_sgs_coefficients_cannot_be_trajectories() -> None:
    prior = _dynamic_prior()
    prior.external_parameters.sgs_bias_b0 = {
        "_target_": "pyurbanair.static_parameters.Normal",
        "mean": 0.0,
        "std": 0.1,
    }
    with pytest.raises(ValueError, match="must be static"):
        augment_sgs_discrepancy_prior(prior, _discrepancy())


def test_filtering_keeps_static_sampler_requirement() -> None:
    cfg = _run_cfg()
    cfg.prior_params = _dynamic_prior()
    with pytest.raises(ValueError, match="static prior sampler"):
        validate_sgs_discrepancy_inference(cfg, "filtering")


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
        ("prior_params._target_", "other", "static ParameterSampler"),
        (
            "esmda.smoother._target_",
            "data_assimilation.smoothing.esmda.StateESMDA",
            "parameter-only",
        ),
        ("params_to_estimate", ["sgs_bias_b0", "sgs_bias_b0"], "unique"),
        ("params_to_estimate", ["unknown_parameter"], "no configured prior"),
        ("params_to_estimate", "sgs_bias_b0", "unique"),
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


@pytest.mark.parametrize(  # type: ignore[misc]
    "selected", [None, list(SGS_BIAS_PARAMETER_NAMES)]
)
@pytest.mark.parametrize("random_walk", [False, True])  # type: ignore[misc]
def test_state_only_filtering_does_not_infer_selected_coefficients(
    selected: Any, random_walk: bool
) -> None:
    cfg = _run_cfg()
    cfg.filtering.mode = "state"
    cfg.params_to_estimate = selected
    if random_walk:
        cfg.filtering.parameter_evolution = {
            "_target_": (
                "data_assimilation.filtering.parameter_evolution.RandomWalkEvolution"
            ),
            "std": 0.1,
        }
    validate_sgs_discrepancy_inference(cfg, "filtering")


@pytest.mark.parametrize("mode", ["parameter", "joint"])  # type: ignore[misc]
def test_filtering_rejects_coefficient_random_walk(mode: str) -> None:
    cfg = _run_cfg()
    cfg.filtering.mode = mode
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
