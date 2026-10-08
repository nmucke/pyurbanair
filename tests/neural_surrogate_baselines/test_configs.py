"""configs/surrogate/baselines/ composes and resolves (tests/scripts/test_configs.py
covers only the top-level surrogate configs)."""

from __future__ import annotations

import pathlib

import pytest
from omegaconf import OmegaConf

from tests.conftest import CONFIGS, compose

BASELINES = CONFIGS / "surrogate" / "baselines"
ENTRY_POINTS = sorted(
    str(p.relative_to(CONFIGS).with_suffix("")) for p in BASELINES.glob("*/*.yaml")
)


def test_there_are_entry_points() -> None:
    assert "surrogate/baselines/local_fno/train" in ENTRY_POINTS


@pytest.mark.parametrize("name", ENTRY_POINTS)  # type: ignore[misc]
def test_entry_point_resolves(name: str, tmp_path: pathlib.Path) -> None:
    cfg = compose(name, root=tmp_path)
    OmegaConf.to_container(cfg, resolve=True)
    # A finetune config inherits its architecture from the pretrained model.
    assert cfg.task == "finetune_stepper" or cfg.architecture._target_.startswith(
        "neural_surrogate_baselines."
    )


def test_compare_resolves(tmp_path: pathlib.Path) -> None:
    cfg = compose("surrogate/baselines/compare", root=tmp_path)
    OmegaConf.to_container(cfg, resolve=True)
