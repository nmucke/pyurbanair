import sys

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

import numpy as np

from pyurbanair.config.hydra_helpers import (
    configure_failure_policy,
    create_observation_operator,
    create_observation_points,
    create_time_varying_true_params,
    create_true_params,
)


def _compose(overrides: list[str] | None = None):
    with initialize(version_base=None, config_path="../conf"):
        return compose(config_name="config", overrides=overrides or [])


@pytest.mark.parametrize(
    "override,expected_name,expected_solver",
    [
        ("model=pylbm", "pylbm", "pylbm"),
        ("model=pyudales", "pyudales", "udales"),
        ("model=pypalm", "pypalm", "palm"),
    ],
)
def test_single_model_configs_compose(
    override: str,
    expected_name: str,
    expected_solver: str,
) -> None:
    cfg = _compose([override])

    assert cfg.model.name == expected_name
    assert cfg.model.solver_name == expected_solver


def test_truth_and_assim_model_aliases_compose() -> None:
    cfg = _compose(["model@truth_model=pylbm", "model@assim_model=pyudales"])

    assert cfg.truth_model.name == "pylbm"
    assert cfg.assim_model.name == "pyudales"
    assert cfg.assim_model.solver_name == "udales"


@pytest.mark.parametrize("preset", ["small", "test"])
def test_presets_compose_with_model_overrides(preset: str) -> None:
    cfg = _compose([f"preset={preset}", "model=pyudales"])

    assert cfg.model.name == "pyudales"
    assert cfg.domain.nx == 40
    assert cfg.obs.mode in {"points", "grid"}


def test_test_preset_matches_fast_test_shape() -> None:
    cfg = _compose(["preset=test"])

    assert cfg.time.simulation_time == 5.0
    assert cfg.time.output_frequency == 1.0
    assert cfg.ensemble.ensemble_size == 4
    assert cfg.ensemble.num_parallel_processes == 1
    assert cfg.obs.mode == "grid"
    assert cfg.obs.aggregation_mode is None
    assert cfg.esmda.num_assimilation_windows == 3
    assert cfg.params.true.inflow_angle == 30.0
    assert True not in cfg.params


def test_palm_target_does_not_import_for_non_palm_composition() -> None:
    for module_name in list(sys.modules):
        if module_name == "pypalm" or module_name.startswith("pypalm."):
            del sys.modules[module_name]

    _compose(["model=pylbm"])

    assert "pypalm" not in sys.modules


def test_interpolations_resolve_under_aliased_packages() -> None:
    cfg = _compose(
        [
            "model@truth_model=pylbm",
            "model@assim_model=pypalm",
            "assim_model.compile=false",
            "esmda.num_steps=4",
        ]
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["assim_model"]["prepare"]["compile"] is False
    assert resolved["esmda"]["alpha"] == 4
    assert resolved["esmda"]["smoother"]["num_steps"] == 4
    assert resolved["esmda"]["smoother"]["alpha"] == 4


def test_true_params_filter_pressure_gradient_for_non_udales() -> None:
    cfg = _compose(["model=pylbm"])

    true_params = create_true_params(cfg.model.name, cfg.params.true)

    assert "pressure_gradient_magnitude" not in true_params


def test_resolve_parameter_schema_includes_pressure_gradient_for_udales() -> None:
    from pyurbanair.config.hydra_helpers import resolve_parameter_schema

    assert resolve_parameter_schema("pylbm") == (
        "inflow_angle",
        "velocity_magnitude",
    )
    assert "pressure_gradient_magnitude" in resolve_parameter_schema("pyudales")


def test_time_varying_truth_and_prior_correlation_lengths_are_distinct() -> None:
    cfg = _compose()

    assert (
        cfg.time_varying.truth_method_kwargs.correlation_length
        != cfg.time_varying.method_kwargs.correlation_length
    )


def _build_time_varying_truth(cfg, num_time_points: int):
    return create_time_varying_true_params(
        model_name=cfg.truth_model.name,
        tv_cfg=cfg.time_varying,
        true_cfg=cfg.params.true,
        external_cfg=cfg.params.external,
        simulation_time=cfg.time.simulation_time,
        num_time_points=num_time_points,
        seed=cfg.esmda.seed,
    )


def test_create_time_varying_true_params_uses_ar2_relaxation_truth() -> None:
    # Default time_varying is ar2_relaxation; truth_method is therefore
    # ar2_relaxation by design. Pin the schema and prove that the truth
    # trajectory is driven by truth_method (not method) by holding
    # truth_method fixed while changing method — the truth must not move.
    cfg_default = _compose(["model@truth_model=pylbm"])
    cfg_other_prior = _compose(
        [
            "model@truth_model=pylbm",
            "time_varying.method=ar1",
            "time_varying.method_kwargs.correlation_length=12345.0",
        ]
    )

    assert cfg_default.time_varying.truth_method == "ar2_relaxation"
    assert cfg_other_prior.time_varying.truth_method == "ar2_relaxation"

    num_time_points = 4
    truth_default = _build_time_varying_truth(cfg_default, num_time_points)
    truth_other = _build_time_varying_truth(cfg_other_prior, num_time_points)

    assert set(truth_default.data_vars) == {"inflow_angle", "velocity_magnitude"}
    assert truth_default["inflow_angle"].dims == ("time",)
    assert truth_default["inflow_angle"].shape == (num_time_points,)
    assert "ensemble" not in truth_default.dims

    np.testing.assert_array_equal(
        truth_default["inflow_angle"].values,
        truth_other["inflow_angle"].values,
    )
    np.testing.assert_array_equal(
        truth_default["velocity_magnitude"].values,
        truth_other["velocity_magnitude"].values,
    )


def test_create_time_varying_true_params_includes_pressure_gradient_for_pyudales() -> None:
    cfg = _compose(["model@truth_model=pyudales"])
    truth = _build_time_varying_truth(cfg, num_time_points=4)

    assert "pressure_gradient_magnitude" in truth
    assert "pressure_gradient_magnitude" not in _build_time_varying_truth(
        _compose(["model@truth_model=pylbm"]), num_time_points=4
    )


def test_observation_helpers_use_explicit_mode() -> None:
    cfg = _compose(["preset=test", "model=pyudales"])

    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)
    obs_op = create_observation_operator(cfg.obs, cfg.model.solver_name)

    assert obs_x.shape == (4,)
    assert obs_y.shape == (4,)
    assert obs_z.shape == (4,)
    assert sorted(zip(obs_x.tolist(), obs_y.tolist(), obs_z.tolist())) == [
        (5.0, 5.0, 2.0),
        (5.0, 35.0, 2.0),
        (35.0, 5.0, 2.0),
        (35.0, 35.0, 2.0),
    ]
    assert obs_op.mode == "mean"
    assert obs_op.observation_operator.num_sensors == 4
    assert obs_op.observation_operator.dim_mapping["u"]["x"] == "xm"


def test_configure_failure_policy_uses_nested_ensemble_config() -> None:
    cfg = _compose(["ensemble.failure.policy=raise", "ensemble.failure.seed=7"])

    class DummyEnsemble:
        def configure_failure_policy(self, policy, jitter_scale, seed):
            self.failure_policy = policy
            self.failure_jitter_scale = jitter_scale
            self.failure_seed = seed

    ensemble = DummyEnsemble()
    returned = configure_failure_policy(ensemble, cfg.ensemble.failure)

    assert returned is ensemble
    assert ensemble.failure_policy == "raise"
    assert ensemble.failure_jitter_scale == cfg.ensemble.failure.jitter_scale
    assert ensemble.failure_seed == 7
