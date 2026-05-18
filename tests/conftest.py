from collections.abc import Sequence
import os

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

os.environ.setdefault("MPLBACKEND", "Agg")


def _compose_test_cfg(overrides: Sequence[str] | None = None) -> DictConfig:
    with initialize(version_base=None, config_path="../conf"):
        return compose(
            config_name="config",
            overrides=["preset=test", *(overrides or [])],
        )


@pytest.fixture
def compose_test_cfg():
    return _compose_test_cfg


@pytest.fixture(scope="module")
def compose_module_cfg():
    """Module-scoped variant of ``compose_test_cfg``.

    Composing inside ``hydra.initialize`` is cheap, but each call still
    opens and closes a ``GlobalHydra`` instance. Module-scoped fixtures
    (e.g. those that compile pylbm once for a whole test module) need a
    composer that can be invoked outside the function-scoped fixture
    lifecycle. This returns the same callable so test code looks
    identical to the function-scoped path.
    """
    return _compose_test_cfg
