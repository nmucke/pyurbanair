"""Production config contracts for the small SGS recovery workflows."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
COEFFICIENTS = ["sgs_bias_b0", "sgs_bias_b1", "sgs_bias_b2"]


def _compose_recipes() -> dict[str, DictConfig]:
    configs = {
        "esmda": ("run_esmda", "esmda/sgs_bias_small"),
        "filtering": ("run_filtering", "filtering/sgs_bias_small"),
        "hybrid": ("run_filter_smoothing", "filter_smoothing/sgs_bias_small"),
    }
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return {
            name: compose(config_name=entrypoint, overrides=[f"experiment={recipe}"])
            for name, (entrypoint, recipe) in configs.items()
        }


def test_small_sgs_recipes_share_the_same_physical_case_and_model() -> None:
    configs = _compose_recipes()
    base = configs["esmda"]

    for cfg in configs.values():
        assert cfg.experiment.workflow in {"esmda", "filtering", "filter_smoothing"}
        assert cfg.run.truth_dir is None
        assert cfg.ensemble.ensemble_size == 12
        assert cfg.ensemble.failure.policy == "raise"
        assert cfg.domain.nx == 32
        assert cfg.domain.ny == 24
        assert cfg.domain.nz == 16
        assert cfg.domain.bounds == [[-16.0, 48.0], [0.0, 48.0], [0.0, 32.0]]
        assert cfg.time.simulation_time == 20.0
        assert cfg.time.output_frequency == 2.0
        assert cfg.time.spinup_time == 10.0
        assert list(cfg.obs.x_points) == list(base.obs.x_points)
        assert list(cfg.obs.y_points) == list(base.obs.y_points)
        assert list(cfg.obs.z_points) == list(base.obs.z_points)
        assert cfg.truth_model.name == cfg.assim_model.name == "pyudales"
        assert cfg.truth_model.forward_model.closure == "vreman"
        assert cfg.truth_model.forward_model.sgs_constant == 0.24
        assert cfg.truth_model.forward_model.model_discrepancy.enabled is True
        assert cfg.truth_model.forward_model.model_discrepancy.prior_std == [
            0.15,
            0.15,
            0.15,
        ]
        assert (
            cfg.prior_params._target_ == "pyurbanair.static_parameters.ParameterSampler"
        )
        assert list(cfg.params_to_estimate) == COEFFICIENTS
        assert (
            OmegaConf.select(
                cfg, "truth_model.forward_model.params.data_vars.inflow_angle"
            )
            == 0.0
        )
        assert (
            OmegaConf.select(
                cfg, "truth_model.forward_model.params.data_vars.velocity_magnitude"
            )
            == 5.0
        )
        assert (
            OmegaConf.select(
                cfg, "assim_model.forward_model.params.data_vars.inflow_angle"
            )
            == 0.0
        )
        assert (
            OmegaConf.select(
                cfg, "assim_model.forward_model.params.data_vars.velocity_magnitude"
            )
            == 5.0
        )

    assert base.truth_params._target_ == "pyurbanair.static_parameters.ParameterSampler"
    assert base.truth_params.parameters.inflow_angle.value == 0.0
    for name, value in zip(COEFFICIENTS, [0.10, -0.12, 0.08]):
        assert base.truth_params.parameters[name].value == value
        assert base.prior_params.parameters[name].std == 0.15
        assert base.prior_params.parameters[name].min is None
        assert base.prior_params.parameters[name].max is None


def test_regular_parameter_configs_contain_static_sgs_coefficients() -> None:
    for name in [
        "static",
        "static_truth",
        "dynamic",
        "dynamic_truth",
        "dynamic_sine",
        "dynamic_cosine",
    ]:
        cfg = OmegaConf.load(CONF_DIR / "params" / f"{name}.yaml")
        block = cfg.parameters if name.startswith("static") else cfg.static_parameters
        assert set(COEFFICIENTS).issubset(block)
    assert not (CONF_DIR / "params/sgs_bias_truth.yaml").exists()


def test_each_small_sgs_workflow_has_distinct_paths_and_algorithm_settings() -> None:
    configs = _compose_recipes()
    esmda, filtering, hybrid = (
        configs["esmda"],
        configs["filtering"],
        configs["hybrid"],
    )

    results_roots = {
        str(OmegaConf.select(cfg, "paths.results_dir")) for cfg in configs.values()
    }
    scratch_roots = {
        str(OmegaConf.select(cfg, "paths.experiment_dir")) for cfg in configs.values()
    }
    assert len(results_roots) == len(scratch_roots) == 3

    assert (
        esmda.esmda.smoother._target_
        == "data_assimilation.smoothing.esmda.ParameterESMDA"
    )
    assert esmda.esmda.num_steps == 3
    assert esmda.esmda.num_assimilation_windows == 1

    assert filtering.filtering.mode == "parameter"
    assert filtering.filtering.num_assimilation_windows == 1
    assert filtering.time.simulation_time / filtering.time.output_frequency == 10

    assert (
        hybrid.esmda.smoother._target_
        == "data_assimilation.smoothing.esmda.ParameterESMDA"
    )
    assert hybrid.esmda.num_steps == 3
    assert hybrid.esmda.num_assimilation_windows == 1
    assert hybrid.filtering.mode == "state"
    assert hybrid.filtering.assimilate_every_n_step == 1
    assert hybrid.filter_smoothing.beta == 2.0
    assert hybrid.filter_smoothing.likelihood_allocation == "shared_budget"
    assert hybrid.esmda.interval_seconds is None
    assert hybrid.observation.aggregation is None
