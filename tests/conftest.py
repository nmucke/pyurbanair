from collections.abc import Sequence

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig


def _compose_test_cfg(overrides: Sequence[str] | None = None) -> DictConfig:
    with initialize(version_base=None, config_path="../conf"):
        return compose(
            config_name="config",
            overrides=["preset=test", *(overrides or [])],
        )


@pytest.fixture
def compose_test_cfg():
    return _compose_test_cfg
