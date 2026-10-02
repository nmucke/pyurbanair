"""configs_new/ composes, resolves and passes the consistency check, and the
test overlays make it tiny."""

from __future__ import annotations

import pathlib

import pytest
from omegaconf import OmegaConf

from tests_new.conftest import CONFIGS, TEST_CONFIGS, compose, load_script

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
VALID: dict[str, list[str]] = {
    "smoother": [],
    "filtering": ["params@prior_params=static"],
    "hybrid": [],
}


@pytest.mark.parametrize("workflow", ["forward", *VALID])  # type: ignore[misc]
def test_check_config_accepts_production(workflow: str, tmp_path: pathlib.Path) -> None:
    check = load_script("scripts_new/inconsistency_check.py").check_config
    name = "forward" if workflow == "forward" else "assimilation"
    check(compose(name, *VALID.get(workflow, []), root=tmp_path), workflow)


@pytest.mark.parametrize(  # type: ignore[misc]
    "workflow, overrides, message",
    [
        ("filtering", [], "static prior_params"),
        ("smoother", ["params@prior_params=static"], "smoothing.smoother is"),
        ("smoother", ["assimilation.params_to_estimate=[nope]"], "does not define"),
    ],
)
def test_check_config_rejects(
    workflow: str, overrides: list[str], message: str, tmp_path: pathlib.Path
) -> None:
    check = load_script("scripts_new/inconsistency_check.py").check_config
    with pytest.raises(ValueError, match=message):
        check(compose("assimilation", *overrides, root=tmp_path), workflow)
