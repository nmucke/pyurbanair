from collections.abc import Sequence
import os

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

os.environ.setdefault("MPLBACKEND", "Agg")


def _compose_test_cfg(
    overrides: Sequence[str] | None = None,
    config_name: str = "config",
) -> DictConfig:
    # ``config_name`` selects the primary config. Forward-model tests use the
    # base ``config``; ESMDA tests use ``run_esmda`` (the single primary config
    # for scripts/run_esmda.py) and pick the smoother via the ``esmda/smoother``
    # group override.
    with initialize(version_base=None, config_path="../conf"):
        return compose(
            config_name=config_name,
            overrides=["+size=test", *(overrides or [])],
        )


@pytest.fixture
def compose_test_cfg():
    return _compose_test_cfg


@pytest.fixture
def surrogate_model_dir_factory():
    """Build a minimal trained-surrogate folder (config.yaml + weights.pt).

    Mirrors what ``scripts/train_neural_surrogate.py`` writes: a model
    ``config.yaml`` holding the architecture and dataset (state_vars /
    param_vars / root_dir), a sibling ``weights.pt`` matching that
    architecture, and a training-data ``config.yaml`` (under ``root_dir``)
    carrying the trained ``domain`` and ``time`` so the forward model can
    derive its trained grid and output frequency. No real data or training
    needed — callers point the surrogate at the returned folder.
    """
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    def _build(
        tmp_path,
        *,
        domain: dict,
        time: dict,
        state_vars=("u", "v", "w"),
        param_vars=("inflow_angle", "velocity_magnitude"),
        architecture: dict | None = None,
    ):
        architecture = architecture or {
            "_target_": "neural_surrogates.UNetConvNeXt",
            "base_channels": 4,
            "channel_mults": [1, 2],
            "depths": [1, 1],
            "kernel_size": 3,
            "expansion": 2,
        }
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
