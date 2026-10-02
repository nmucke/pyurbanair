"""Fixtures of the library tests, carried over unchanged from tests/conftest.py.

They compose the frozen test configs in legacy/conf/ (the old conf/ schema).
Registered in tests/conftest.py. To be replaced by configs overlays
(tests/configs/) as the library tests are rewritten.
"""

import itertools
import os
import pathlib
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence

# On macOS the native loader can abort if torch initializes before NumPy.
# Import NumPy first so ordinary pytest collection works without a wrapper.
import numpy  # noqa: F401
import pytest
from omegaconf import DictConfig

from tests.legacy.config_loader import compose_test_config

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Physical, solver, and resource settings are fixed in tests/conf. This fixture
# only supplies unique scratch paths and caller overrides; production configs
# do not participate in test composition.
_OUTPUT_ROOT: pathlib.Path | None = None
_RUN_COUNTER = itertools.count()
_LBM_BUILD_CACHE = (
    pathlib.Path(tempfile.gettempdir()) / f"pyurbanair-pytest-lbm-build-{os.getuid()}"
)


def _output_root() -> pathlib.Path:
    global _OUTPUT_ROOT
    if _OUTPUT_ROOT is None:
        _OUTPUT_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="pyurbanair-tests-"))
    return _OUTPUT_ROOT


@pytest.fixture(scope="session", autouse=True)  # type: ignore[misc]
def _isolate_run_outputs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    global _OUTPUT_ROOT
    if _OUTPUT_ROOT is None:
        _OUTPUT_ROOT = tmp_path_factory.mktemp("run_outputs")
    os.environ.setdefault("PYLBM_BUILD_ROOT", str(_LBM_BUILD_CACHE))
    yield


@pytest.fixture(scope="session", autouse=True)  # type: ignore[misc]
def _limit_torch_test_threads() -> Iterator[None]:
    """Keep small CPU network tests from oversubscribing the test host."""
    torch = sys.modules.get("torch")
    if torch is None:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _isolated_path_overrides() -> list[str]:
    run_dir = _output_root() / f"run_{next(_RUN_COUNTER):04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        f"paths.results_dir={run_dir / 'results'}",
        f"paths.experiment_dir={run_dir / 'experiment'}",
        f"++paths.base_results_dir={run_dir / 'base_results'}",
    ]


def _compose_test_cfg(
    overrides: Sequence[str] | None = None,
    config_name: str = "run_forward_model",
) -> DictConfig:
    # Place caller overrides last so tests can deliberately opt into a feature
    # or choose their own temporary path.
    return compose_test_config(
        [*_isolated_path_overrides(), *list(overrides or ())],
        config_name=config_name,
    )


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _restore_hydra_config_singleton() -> Iterator[None]:
    """Keep ``HydraConfig`` from leaking a composed config across test files.

    ``HydraConfig`` is a process-wide singleton, so a test that primes it with
    ``HydraConfig.instance().set_config(cfg)`` leaves it populated for the rest
    of the session. A config from bare ``compose()`` has no
    ``hydra.runtime.output_dir`` (it is ``???``), so any later test that reaches
    ``resolve_output_dir`` takes its ``HydraConfig.initialized()`` branch and
    dies on MissingMandatoryValue instead of falling back to
    ``paths.base_results_dir``. Whether that happens comes down to file
    collection order, which makes it a nasty failure to place.
    """
    from hydra.core.hydra_config import HydraConfig

    previous = HydraConfig.instance().cfg
    yield
    HydraConfig.instance().cfg = previous


@pytest.fixture  # type: ignore[misc]
def compose_test_cfg() -> Callable[..., DictConfig]:
    return _compose_test_cfg


@pytest.fixture  # type: ignore[misc]
def surrogate_model_dir_factory() -> Callable[..., pathlib.Path]:
    """Build a minimal trained-surrogate folder (config.yaml + weights.pt).

    Mirrors what ``scripts/neural_surrogate/train_neural_surrogate.py`` writes: a model
    ``config.yaml`` holding the architecture and dataset (state_vars /
    param_vars / root_dir), a sibling ``weights.pt`` matching that
    architecture, and a training-data ``config.yaml`` (under ``root_dir``)
    carrying the trained ``domain`` and ``time`` so the forward model can
    derive its trained grid and output frequency. No real data or training
    needed — callers point the surrogate at the returned folder.

    ``num_history_steps`` (how many past frames the network consumes) is only
    written — under both ``architecture`` and ``dataset``, as the trainer does —
    when it differs from the default of 1, so the default folder is unchanged.
    """
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    def _build(
        tmp_path: pathlib.Path,
        *,
        domain: dict,
        time: dict,
        state_vars: Sequence[str] = ("u", "v", "w"),
        param_vars: Sequence[str] = ("inflow_angle", "velocity_magnitude"),
        architecture: dict | None = None,
        num_history_steps: int = 1,
    ) -> pathlib.Path:
        architecture = architecture or {
            "_target_": "neural_surrogates.UNetConvNeXt",
            "base_channels": 4,
            "channel_mults": [1, 2],
            "depths": [1, 1],
            "kernel_size": 3,
            "expansion": 2,
        }
        # A one-step surrogate (the default) writes exactly the config the
        # trainer has always written; the key only appears for H > 1.
        if num_history_steps != 1:
            architecture = {**architecture, "num_history_steps": num_history_steps}
        root_dir = tmp_path / "training_data"
        root_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create({"domain": domain, "time": time}),
            root_dir / "config.yaml",
        )

        model_dir = tmp_path / "model_dir"
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "architecture": architecture,
                    "dataset": {
                        "root_dir": str(root_dir),
                        "state_vars": list(state_vars),
                        "param_vars": list(param_vars),
                        **(
                            {}
                            if num_history_steps == 1
                            else {"num_history_steps": num_history_steps}
                        ),
                    },
                }
            ),
            model_dir / "config.yaml",
        )
        model = instantiate(
            architecture,
            n_state_channels=len(state_vars),
            n_params=len(param_vars),
        )
        torch.save(model.state_dict(), model_dir / "weights.pt")
        return model_dir

    return _build


@pytest.fixture(scope="module")  # type: ignore[misc]
def compose_module_cfg() -> Callable[..., DictConfig]:
    """Module-scoped variant of ``compose_test_cfg``.

    Composing inside ``hydra.initialize`` is cheap, but each call still
    opens and closes a ``GlobalHydra`` instance. Module-scoped fixtures
    (e.g. those that compile pylbm once for a whole test module) need a
    composer that can be invoked outside the function-scoped fixture
    lifecycle. This returns the same callable so test code looks
    identical to the function-scoped path.
    """
    return _compose_test_cfg
