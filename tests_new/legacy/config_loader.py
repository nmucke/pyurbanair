"""Compose the independent, frozen configurations used by integration tests."""

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

TEST_CONF_DIR = Path(__file__).resolve().parent / "conf"


def compose_test_config(
    overrides: Sequence[str] | None = None,
    *,
    config_name: str = "run_forward_model",
) -> DictConfig:
    """Compose from tests/conf only, without reading a production run config."""
    with initialize_config_dir(version_base=None, config_dir=str(TEST_CONF_DIR)):
        return compose(config_name=config_name, overrides=list(overrides or ()))
