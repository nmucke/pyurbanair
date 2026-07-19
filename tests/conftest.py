import os
from collections.abc import Sequence

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

os.environ.setdefault("MPLBACKEND", "Agg")

# Test smoke shape: the smallest, fastest run the solvers accept — a tiny
# [0,20]^2 x [0,10] domain, a 3 s window and a 2-member ensemble. Applied to
# every composed test config so the suite exercises the code paths without
# producing a meaningful flow (formerly the deleted `+scale=test` overlay).
_SMOKE_OVERRIDES = [
    "domain.nx=20",
    "domain.ny=20",
    "domain.nz=4",
    "domain.bounds=[[0.0,20.0],[0.0,20.0],[0.0,10.0]]",
    "time.simulation_time=3.0",
    "time.output_frequency=1.0",
    "time.spinup_time=3.0",
    "time.seconds_per_knot=1.5",
    "ensemble.ensemble_size=2",
    "ensemble.num_parallel_processes=1",
]


# run_esmda.yaml ships production defaults the suite must not inherit:
# machine-specific scratch roots (/export/...) instead of the portable in-repo
# roots (the commented-out defaults in that file), and ``case: barcelona``,
# whose precomputed uDALES geometry bundle only matches the Barcelona grid —
# not the smoke domain above. Re-assert the test-friendly xie_and_castro case
# (the default of the other entry points) and the in-repo output roots.
_ESMDA_OVERRIDES = [
    "case=xie_and_castro",
    "paths.results_dir=.temp/${truth_model.name}_to_${assim_model.name}",
    "paths.experiment_dir=${oc.env:PWD}/.temp",
]


def _compose_test_cfg(
    overrides: Sequence[str] | None = None,
    config_name: str = "run_forward_model",
) -> DictConfig:
    # ``config_name`` selects the primary config (entry point). Forward-model
    # tests use ``run_forward_model``; ESMDA tests use ``run_esmda`` (the single
    # primary config for scripts/esmda/run_esmda.py) and pick the smoother via the
    # ``esmda/smoother`` group override.
    esmda_overrides = _ESMDA_OVERRIDES if config_name == "run_esmda" else []
    with initialize(version_base=None, config_path="../conf"):
        return compose(
            config_name=config_name,
            overrides=[*_SMOKE_OVERRIDES, *esmda_overrides, *(overrides or [])],
        )


@pytest.fixture
def compose_test_cfg():
    return _compose_test_cfg


@pytest.fixture
def surrogate_model_dir_factory():
    """Build a minimal trained-surrogate folder (config.yaml + weights.pt).

    Mirrors what ``scripts/neural_surrogate/train_neural_surrogate.py`` writes: a model
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
