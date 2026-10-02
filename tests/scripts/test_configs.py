"""configs/ composes, resolves and passes the consistency check, and the
test overlays make it tiny."""

from __future__ import annotations

import pathlib

import pytest
from omegaconf import OmegaConf

from tests.conftest import CONFIGS, TEST_CONFIGS, compose, load_script

ENTRY_POINTS = [
    "forward",
    "assimilation",
    *(f"surrogate/{p.stem}" for p in sorted((CONFIGS / "surrogate").glob("*.yaml"))),
]
ENTRY_POINTS.remove("surrogate/architectures")  # entries, not a run config
ENTRY_POINTS.remove("surrogate/training")  # shared defaults, not a run config
# Every option of the groups forward.yaml picks from, incl. the tiny test backends.
OPTIONS = {
    group: sorted(
        p.stem for d in (CONFIGS, TEST_CONFIGS) for p in (d / group).glob("*.yaml")
    )
    for group in ("model", "params", "case")
}


@pytest.mark.parametrize("name", ENTRY_POINTS)  # type: ignore[misc]
def test_entry_point_resolves(name: str, tmp_path: pathlib.Path) -> None:
    OmegaConf.to_container(compose(name, root=tmp_path), resolve=True)


@pytest.mark.parametrize("group", OPTIONS)  # type: ignore[misc]
def test_every_option_resolves(group: str, tmp_path: pathlib.Path) -> None:
    for option in OPTIONS[group]:
        cfg = compose("forward", f"{group}={option}", root=tmp_path)
        OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize(  # type: ignore[misc]
    "name", ["forward", "assimilation", "surrogate/generate_data"]
)
def test_test_overlay_is_tiny(name: str, tmp_path: pathlib.Path) -> None:
    overlay = "+test=" + name.split("/")[-1]
    cfg = compose(name, overlay, root=tmp_path)
    assert (cfg.domain.nx, cfg.domain.ny, cfg.domain.nz) == (20, 20, 4)
    assert cfg.ensemble.ensemble_size == 2


# Workflow -> overrides that make the production assimilation config valid for it.
# filtering.mode=state needs filtering.parameter_evolution=null; set it here so the
# test doesn't depend on the value in configs/assimilation.yaml.
VALID: dict[str, list[str]] = {
    "smoother": [],
    "filtering": ["params@prior_params=static", "filtering.parameter_evolution=null"],
    "hybrid": ["filtering.parameter_evolution=null"],
}


@pytest.mark.parametrize("workflow", ["forward", *VALID])  # type: ignore[misc]
def test_check_config_accepts_production(workflow: str, tmp_path: pathlib.Path) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    name = "forward" if workflow == "forward" else "assimilation"
    check(compose(name, *VALID.get(workflow, []), root=tmp_path), workflow)


@pytest.mark.parametrize(  # type: ignore[misc]
    "workflow, overrides, message",
    [
        ("filtering", [], "static prior_params"),
        ("smoother", ["params@prior_params=static"], "smoothing.smoother is"),
        ("smoother", ["assimilation.params_to_estimate=[nope]"], "does not define"),
        (
            "smoother",
            ["domain.nx=30", "assim_model.forward_model.ncpu=7"],
            "must divide nx",
        ),
        (
            "smoother",
            [
                "model@assim_model=pypalm",
                "assim_model.forward_model.boundary_condition=periodic",
                "domain.nx=31",
            ],
            "even nx and ny",
        ),
    ],
)
def test_check_config_rejects(
    workflow: str, overrides: list[str], message: str, tmp_path: pathlib.Path
) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    with pytest.raises(ValueError, match=message):
        check(compose("assimilation", *overrides, root=tmp_path), workflow)


# A valid smoother run estimating the SGS-discrepancy coefficients; set
# explicitly so the test doesn't depend on configs/assimilation.yaml's tuning.
DISCREPANCY = [
    "assim_model.forward_model.model_discrepancy.enabled=true",
    "params@truth_params=dynamic_sine",
    "params@prior_params=dynamic",
    "smoothing.smoother=${smoother.dynamic}",
    "smoothing.localization=null",
    "smoothing.state_reduction=null",
    "smoothing.final_time_smoothing=false",
    "assimilation.params_to_estimate=null",
    "filtering.mode=state",
    "filtering.parameter_evolution=null",
    "ensemble.failure.policy=raise",
]
STATIC_COEFFICIENT_FILTER = [
    "params@prior_params=static",
    "filtering.mode=parameter",
    "assimilation.params_to_estimate=[sgs_bias_b0]",
]


def test_check_config_accepts_discrepancy(tmp_path: pathlib.Path) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    check(compose("assimilation", *DISCREPANCY, root=tmp_path), "smoother")


@pytest.mark.parametrize(  # type: ignore[misc]
    "workflow, overrides, message",
    [
        ("smoother", ["assim_model.name=pylbm"], "needs pyudales with vreman"),
        (
            "smoother",
            ["assim_model.forward_model.closure=smagorinsky"],
            "needs pyudales with vreman",
        ),
        (
            "smoother",
            [
                "+prior_params.external_parameters.sgs_bias_b0="
                "{_target_:pyurbanair.static_parameters.Normal,mean:0.0,std:0.1}"
            ],
            "must be static parameters",
        ),
        (
            "smoother",
            ["smoothing.smoother=${smoother.state_and_dynamic}"],
            "smoother must be TimeVaryingParameterESMDA",
        ),
        ("hybrid", ["filtering.mode=joint"], "needs filtering.mode=state"),
        (
            "hybrid",
            ["ensemble.failure.policy=resample_from_successes"],
            "needs ensemble.failure.policy=raise",
        ),
        (
            "filtering",
            [
                *STATIC_COEFFICIENT_FILTER,
                "filtering.parameter_evolution={_target_:data_assimilation."
                "filtering.parameter_evolution.RandomWalkEvolution,std:0.1}",
            ],
            "parameter_evolution=null",
        ),
        (
            "filtering",
            [
                *STATIC_COEFFICIENT_FILTER,
                "filtering.localization=${localization.distance}",
            ],
            "cannot use distance localization",
        ),
    ],
)
def test_check_config_rejects_discrepancy(
    workflow: str, overrides: list[str], message: str, tmp_path: pathlib.Path
) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    cfg = compose("assimilation", *DISCREPANCY, *overrides, root=tmp_path)
    with pytest.raises(ValueError, match=message):
        check(cfg, workflow)
