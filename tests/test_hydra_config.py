import builtins
import io
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from data_assimilation.observation_operator import TemporalObservationOperator
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pyurbanair.config.hydra_helpers import (
    create_aggregate_observations,
    create_observation_operator,
    create_observation_points,
)
from tests.config_loader import TEST_CONF_DIR, compose_test_config


def _compose(
    overrides: list[str] | None = None, config_name: str = "run_forward_model"
) -> Any:
    return compose_test_config(overrides, config_name=config_name)


@pytest.mark.parametrize(  # type: ignore[misc]
    "override,expected_name,expected_solver",
    [
        ("model=pylbm", "pylbm", "pylbm"),
        ("model=pyudales", "pyudales", "udales"),
        ("model=pypalm", "pypalm", "palm"),
    ],
)
def test_single_model_configs_compose(
    override: Any, expected_name: Any, expected_solver: Any
) -> None:
    cfg = _compose([override])
    assert cfg.model.name == expected_name
    assert cfg.model.solver_name == expected_solver


def test_truth_and_assim_model_aliases_compose() -> None:
    cfg = _compose(
        ["model@truth_model=pylbm", "model@assim_model=pyudales"],
        config_name="run_esmda",
    )
    assert cfg.truth_model.name == "pylbm"
    assert cfg.assim_model.name == "pyudales"
    assert cfg.assim_model.solver_name == "udales"


@pytest.mark.parametrize(  # type: ignore[misc]
    "config_name,mounts",
    [
        ("run_forward_model", ("model",)),
        ("run_esmda", ("truth_model", "assim_model")),
        ("run_filtering", ("truth_model", "assim_model")),
        ("run_filter_smoothing", ("truth_model", "assim_model")),
    ],
)
def test_fixed_udales_smoke_settings(config_name: Any, mounts: Any) -> None:
    cfg = _compose(config_name=config_name)
    for mount in mounts:
        model = cfg[mount]
        assert model.forward_model.ncpu == 1
        assert model.forward_model.inlet_turbulence.enabled is False
        assert model.forward_model.nudging_config.nnudge_meters == 4.0


def test_caller_override_wins_over_test_baseline() -> None:
    cfg = _compose(
        [
            "model.forward_model.ncpu=2",
            "model.forward_model.inlet_turbulence.enabled=true",
        ]
    )
    assert cfg.model.forward_model.ncpu == 2
    assert cfg.model.forward_model.inlet_turbulence.enabled is True


def test_fixed_test_config_is_independent_of_production_run_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A conflicting cwd/conf tree cannot change the frozen smoke configuration.
    shadow_conf = tmp_path / "conf"
    shadow_conf.mkdir()
    (shadow_conf / "run_forward_model.yaml").write_text(
        "domain:\n  nx: 999\nensemble:\n  ensemble_size: 999\n"
    )
    monkeypatch.chdir(tmp_path)
    assert TEST_CONF_DIR != Path(__file__).resolve().parents[1] / "conf"
    for path in TEST_CONF_DIR.rglob("*.yaml"):
        assert "hydra.searchpath" not in path.read_text()
    with initialize_config_dir(version_base=None, config_dir=str(TEST_CONF_DIR)):
        composed = compose(config_name="run_forward_model", return_hydra_config=True)
    main_sources = [
        source.path
        for source in composed.hydra.runtime.config_sources
        if source.provider == "main"
    ]
    assert main_sources == [str(TEST_CONF_DIR)]
    production_conf = Path(__file__).resolve().parents[1] / "conf"
    original_open = builtins.open
    original_io_open = io.open

    def forbid_production_config_reads(opener: Any) -> Any:
        def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, os.PathLike)):
                path = Path(file).resolve()
                if path.is_relative_to(production_conf):
                    raise AssertionError(
                        f"test composition read production config: {path}"
                    )
            return opener(file, *args, **kwargs)

        return guarded

    monkeypatch.setattr(builtins, "open", forbid_production_config_reads(original_open))
    monkeypatch.setattr(io, "open", forbid_production_config_reads(original_io_open))
    for name in (
        "run_forward_model",
        "run_esmda",
        "run_filtering",
        "run_filter_smoothing",
    ):
        cfg = _compose(config_name=name)
        assert (cfg.domain.nx, cfg.domain.ny, cfg.domain.nz) == (20, 20, 4)
        assert cfg.time.simulation_time == 3.0
        assert cfg.time.output_frequency == 1.0
        assert cfg.ensemble.ensemble_size == 2
        assert cfg.ensemble.num_parallel_processes == 1
    data = _compose(config_name="neural_surrogate/training_data")
    assert data.training_data.simulation_time == 3.0
    assert data.training_data.adaptive_spinup.enabled is False
    assert data.training_data.num_train == 2
    fixed_data = _compose(
        ["training_data/geometry_mode=fixed", "case=barcelona"],
        config_name="neural_surrogate/training_data",
    )
    assert fixed_data.training_data.geometry.output_name == "barcelona"


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
        ],
        config_name="run_esmda",
    )
    resolved = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    assert resolved["assim_model"]["prepare"]["compile"] is False
    assert resolved["esmda"]["alpha"] == 4
    assert resolved["esmda"]["smoother"]["num_steps"] == 4
    assert resolved["esmda"]["smoother"]["alpha"] == 4


def test_observation_components_use_test_group_targets() -> None:
    cfg = _compose(config_name="run_esmda")
    assert cfg.observation.operator._target_.endswith("TemporalObservationOperator")
    op = create_observation_operator(
        cfg.obs, cfg.truth_model.solver_name, cfg.observation.operator
    )
    aggregation = create_aggregate_observations(cfg)
    assert isinstance(op, TemporalObservationOperator)
    assert op.observation_operator.num_sensors == len(cfg.obs.x_points)
    assert aggregation is not None
    assert aggregation.interval_seconds == cfg.esmda.interval_seconds


def test_legacy_observation_fallback_without_group() -> None:
    cfg = _compose()
    assert "observation" not in cfg
    op = create_observation_operator(cfg.obs, cfg.model.solver_name)
    assert isinstance(op, TemporalObservationOperator)


def test_resolve_parameter_schema_includes_pressure_gradient_for_udales() -> None:
    from pyurbanair.config.hydra_helpers import resolve_parameter_schema

    assert resolve_parameter_schema("pylbm") == (
        "inflow_angle",
        "velocity_magnitude",
        "vertical_inflow_exponent",
        "sgs_constant",
    )
    assert "pressure_gradient_magnitude" in resolve_parameter_schema("pyudales")


def test_resolve_parameter_schema_includes_model_error_knobs() -> None:
    """Every backend advertises the two model-error compensation knobs."""
    from pyurbanair.config.hydra_helpers import resolve_parameter_schema

    for model in ("pylbm", "pyudales", "pypalm"):
        schema = resolve_parameter_schema(model)
        assert "vertical_inflow_exponent" in schema
        assert "sgs_constant" in schema


def test_filter_parameter_config_restricts_static_sampler() -> None:
    """Explicit sampler pruning remains available outside DA selection."""
    from omegaconf import OmegaConf

    from pyurbanair.config.hydra_helpers import filter_parameter_config

    full = OmegaConf.create(
        {
            "parameters": {
                "inflow_angle": {"a": 1},
                "velocity_magnitude": {"a": 1},
                "vertical_inflow_exponent": {"a": 1},
                "sgs_constant": {"a": 1},
            }
        }
    )

    # null -> unchanged (all parameters kept).
    assert set(filter_parameter_config(full, None)["parameters"]) == set(
        full["parameters"]
    )

    # subset -> only the named parameters survive; original is not mutated.
    selected = ["inflow_angle", "velocity_magnitude"]
    pruned = filter_parameter_config(full, selected)
    assert set(pruned["parameters"]) == set(selected)
    assert "sgs_constant" in full["parameters"]


def test_filter_parameter_config_restricts_dynamic_sampler() -> None:
    """The filter prunes BOTH the dynamic and static blocks of the AR(2) config."""
    from omegaconf import OmegaConf

    from pyurbanair.config.hydra_helpers import filter_parameter_config

    full = OmegaConf.create(
        {
            "external_parameters": {
                "inflow_angle": {"a": 1},
                "velocity_magnitude": {"a": 1},
            },
            "static_parameters": {
                "vertical_inflow_exponent": {"a": 1},
                "sgs_constant": {"a": 1},
            },
        }
    )

    # Keep one dynamic + one static parameter; drop the rest.
    pruned = filter_parameter_config(full, ["inflow_angle", "sgs_constant"])
    assert set(pruned["external_parameters"]) == {"inflow_angle"}
    assert set(pruned["static_parameters"]) == {"sgs_constant"}


@pytest.mark.parametrize("kind", ["static", "dynamic"])  # type: ignore[misc]
@pytest.mark.parametrize("selected", [None, [], ["inflow_angle"]])  # type: ignore[misc]
def test_inference_parameter_selection_preserves_full_samplers(
    kind: str, selected: Any
) -> None:
    import xarray
    from hydra.utils import instantiate

    from pyurbanair.config.hydra_helpers import inference_parameter_configs

    cfg = _compose(
        [f"params@prior_params={kind}", f"params@truth_params={kind}_truth"],
        config_name="run_esmda",
    )
    cfg.params_to_estimate = selected
    original = OmegaConf.to_container(cfg, resolve=True)

    truth_cfg, prior_cfg = inference_parameter_configs(cfg)
    assert OmegaConf.to_container(cfg, resolve=True) == original
    xarray.testing.assert_identical(
        instantiate(truth_cfg).sample(1), instantiate(cfg.truth_params).sample(1)
    )
    xarray.testing.assert_identical(
        instantiate(prior_cfg).sample(4), instantiate(cfg.prior_params).sample(4)
    )
    # The unselected shear parameter must still be applied to both models.
    assert "vertical_inflow_exponent" in instantiate(truth_cfg).sample(1)
    assert "vertical_inflow_exponent" in instantiate(prior_cfg).sample(4)


@pytest.mark.parametrize(  # type: ignore[misc]
    "selected,error",
    [
        (["unknown_parameter"], "no configured prior"),
        (["inflow_angle", "inflow_angle"], "unique"),
        ("inflow_angle", "unique"),
        ([""], "unique"),
        ([1], "unique"),
    ],
)
def test_inference_parameter_selection_rejects_invalid_names(
    selected: Any, error: str
) -> None:
    from pyurbanair.config.hydra_helpers import inference_parameter_configs

    cfg = _compose(config_name="run_esmda")
    cfg.params_to_estimate = selected
    with pytest.raises(ValueError, match=error):
        inference_parameter_configs(cfg)


def test_observation_helpers_use_explicit_mode() -> None:
    # Grid-mode obs supplied explicitly (the helpers dispatch on obs.mode); a
    # 2x2 grid over [5,35]^2 at z=2 yields the four corner sensors below.
    cfg = _compose(
        [
            "model=pyudales",
            "obs.mode=grid",
            "+obs.x_min=5.0",
            "+obs.x_max=35.0",
            "+obs.y_min=5.0",
            "+obs.y_max=35.0",
            "+obs.n_per_axis=2",
            "+obs.z=2.0",
            "obs.temporal_mode=full",
        ]
    )

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
    assert isinstance(obs_op, TemporalObservationOperator)
    assert obs_op.observation_operator.num_sensors == 4
    assert obs_op.observation_operator.dim_mapping["u"]["x"] == "xm"
