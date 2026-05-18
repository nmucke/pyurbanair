import sys

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from pyurbanair.config.hydra_helpers import create_true_params


def _compose(overrides: list[str] | None = None):
    with initialize(version_base=None, config_path="../conf"):
        return compose(config_name="config", overrides=overrides or [])


@pytest.mark.parametrize(
    "override,expected_name,expected_solver,expected_init_subdir",
    [
        ("model=pylbm", "pylbm", "pylbm", "lbm"),
        ("model=pyudales", "pyudales", "udales", "udales"),
        ("model=pypalm", "pypalm", "palm", "palm"),
    ],
)
def test_single_model_configs_compose(
    override: str,
    expected_name: str,
    expected_solver: str,
    expected_init_subdir: str,
) -> None:
    cfg = _compose([override])

    assert cfg.model.name == expected_name
    assert cfg.model.solver_name == expected_solver
    assert cfg.model.init_subdir == expected_init_subdir


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


def test_time_varying_truth_and_prior_correlation_lengths_are_distinct() -> None:
    cfg = _compose()

    assert (
        cfg.time_varying.truth_method_kwargs.correlation_length
        != cfg.time_varying.method_kwargs.correlation_length
    )
