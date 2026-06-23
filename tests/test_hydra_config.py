import sys

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from pyurbanair.config.hydra_helpers import (
    create_observation_operator,
    create_observation_points,
)


def _compose(overrides: list[str] | None = None):
    with initialize(version_base=None, config_path="../conf"):
        return compose(config_name="run_forward_model", overrides=overrides or [])


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
    # The truth/assim double-mount lives in the run_esmda entry point.
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="run_esmda",
            overrides=["model@truth_model=pylbm", "model@assim_model=pyudales"],
        )

    assert cfg.truth_model.name == "pylbm"
    assert cfg.assim_model.name == "pyudales"
    assert cfg.assim_model.solver_name == "udales"


def test_entrypoint_composes_with_model_override() -> None:
    cfg = _compose(["model=pyudales"])

    assert cfg.model.name == "pyudales"
    # Physics comes from the default case (xie_and_castro).
    assert cfg.domain.nx == 60
    assert cfg.obs.mode in {"points", "grid"}


def test_default_compute_is_medium() -> None:
    # The former scale=medium values are now baked straight into the entry point.
    cfg = _compose([])

    assert cfg.ensemble.ensemble_size == 64
    assert cfg.ensemble.num_parallel_processes == 4
    assert cfg.time.seconds_per_knot == 45.0


def test_palm_target_does_not_import_for_non_palm_composition() -> None:
    for module_name in list(sys.modules):
        if module_name == "pypalm" or module_name.startswith("pypalm."):
            del sys.modules[module_name]

    _compose(["model=pylbm"])

    assert "pypalm" not in sys.modules


def test_interpolations_resolve_under_aliased_packages() -> None:
    # The smoother is supplied by the run_esmda primary config's
    # ``esmda/smoother`` group, so compose it (rather than the base ``config``)
    # to exercise the smoother interpolations.
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="run_esmda",
            overrides=[
                "model@truth_model=pylbm",
                "model@assim_model=pypalm",
                "assim_model.compile=false",
                "esmda.num_steps=4",
            ],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["assim_model"]["prepare"]["compile"] is False
    assert resolved["esmda"]["alpha"] == 4
    assert resolved["esmda"]["smoother"]["num_steps"] == 4
    assert resolved["esmda"]["smoother"]["alpha"] == 4


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
    """`params_to_estimate` prunes the static sampler's `parameters` block."""
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
            "obs.temporal_mode=mean",
            "obs.aggregation_mode=null",
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
    assert obs_op.mode == "mean"
    assert obs_op.observation_operator.num_sensors == 4
    assert obs_op.observation_operator.dim_mapping["u"]["x"] == "xm"
